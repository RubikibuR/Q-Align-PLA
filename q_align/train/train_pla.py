# PLA Training Script
# Adapted from Q-Align train_mem.py for Protein-Ligand Affinity Prediction
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch
import transformers
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error

from torch.utils.data import Dataset
from q_align.train.mplug_owl2_trainer import MPLUGOwl2Trainer, PLATrainer
from q_align.constants import IGNORE_INDEX, PROTEIN_TOKEN_INDEX, LIGAND_TOKEN_INDEX

from q_align import conversation as conversation_lib
from q_align.model import *
from q_align.train.pla_dataset import PLADataset, PLADataCollator, calculate_affinity_levels
from q_align.constants import PLA_LEVEL_NAMES

from icecream import ic
import numpy as np

local_rank = None


def get_preferential_ids(tokenizer) -> List[int]:
    """返回 PLA_LEVEL_NAMES 中每个词在词表中的 token ID（取 BOS 之后的第一个 token）。"""
    return [ids[1] for ids in tokenizer(PLA_LEVEL_NAMES)["input_ids"]]


def preds_to_classes(values, thresholds):
    """将连续亲和力值按 thresholds 映射为类别索引（0 ~ num_levels-1）。"""
    # thresholds 有 num_levels+1 个边界，区间 [thresholds[i], thresholds[i+1])
    classes = np.zeros(len(values), dtype=np.int64)
    for i in range(len(thresholds) - 1):
        if i == len(thresholds) - 2:
            # 最后一个区间包含右边界
            mask = values >= thresholds[i]
        else:
            mask = (values >= thresholds[i]) & (values < thresholds[i + 1])
        classes[mask] = i
    return classes


def compute_affinity_metrics(predictions, labels, thresholds=None) -> dict:
    """计算 RMSE / MAE / Pearson R 等指标，返回指标字典。

    predictions: shape [N, 2] — 第0列=pred_pkd, 第1列=pred_class (argmax)
                 或 shape [N] — 仅 pred_pkd（向后兼容）
    labels:      shape [N] — 真实 pKd 值
    """
    predictions = np.array(predictions, dtype=np.float64)
    lbls = np.array(labels, dtype=np.float64).flatten()

    # 解包 predictions
    if predictions.ndim == 2 and predictions.shape[1] >= 2:
        preds = predictions[:, 0].flatten()
        pred_classes = predictions[:, 1].flatten().astype(np.int64)
    else:
        preds = predictions.flatten()
        pred_classes = None

    valid = np.isfinite(preds) & np.isfinite(lbls)
    preds, lbls = preds[valid], lbls[valid]
    if pred_classes is not None:
        pred_classes = pred_classes[valid]

    if len(preds) == 0:
        return {
            "rmse": float("inf"),
            "mae": float("inf"),
            "sd": float("inf"),
            "pearson_r": 0.0,
            "pearson_p": 1.0,
            "spearman_r": 0.0,
            "spearman_p": 1.0,
            "accuracy": 0.0,
            "num_samples": 0,
        }

    rmse = float(np.sqrt(mean_squared_error(lbls, preds)))
    mae = float(mean_absolute_error(lbls, preds))
    sd = float(np.std(preds - lbls, ddof=1)) if len(preds) > 1 else 0.0

    if len(preds) > 1:
        pearson_r, pearson_p = pearsonr(lbls, preds)
        spearman_r, spearman_p = spearmanr(lbls, preds)
    else:
        pearson_r, pearson_p = 0.0, 1.0
        spearman_r, spearman_p = 0.0, 1.0

    accuracy = 0.0
    if thresholds is not None:
        true_classes = preds_to_classes(lbls, thresholds)
        if pred_classes is not None:
            # 直接使用 argmax 类别计算 accuracy
            accuracy = float(np.mean(pred_classes == true_classes))
        else:
            # 向后兼容：从连续预测值回映射
            pred_classes_from_pkd = preds_to_classes(preds, thresholds)
            accuracy = float(np.mean(pred_classes_from_pkd == true_classes))

    return {
        "rmse": rmse,
        "mae": mae,
        "sd": sd,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "accuracy": accuracy,
        "num_samples": int(len(preds)),
    }


def make_compute_pla_metrics(thresholds):
    """返回绑定了 thresholds 的 compute_metrics 回调。"""
    def compute_pla_metrics(eval_preds):
        metrics = compute_affinity_metrics(eval_preds.predictions, eval_preds.label_ids, thresholds=thresholds)
        rank0_print(
            f"  [Val] RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  "
            f"SD={metrics['sd']:.4f}  Pearson R={metrics['pearson_r']:.4f}  "
            f"Accuracy={metrics['accuracy']:.4f}"
        )
        return {k: metrics[k] for k in ("rmse", "mae", "sd", "pearson_r", "accuracy")}
    return compute_pla_metrics


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="vicuna_v1")
    freeze_backbone: bool = field(default=False)

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training CSV file."})
    val_data_path: Optional[str] = field(default=None,
                           metadata={"help": "Path to the validation CSV file."})
    test_data_path: Optional[str] = field(default=None,
                           metadata={"help": "Path to the test CSV file for final evaluation."})
    lazy_preprocess: bool = False
    is_multimodal: bool = True  # PLA is always multimodal
    num_affinity_levels: int = field(default=5,
                           metadata={"help": "Number of affinity levels (3 or 5)."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="accuracy")
    greater_is_better: bool = field(default=True)

    # PLA specific arguments
    tune_protein_abstractor: bool = field(default=True)
    tune_ligand_abstractor: bool = field(default=True)
    freeze_protein_encoder: bool = field(default=True)
    freeze_ligand_encoder: bool = field(default=True)
    freeze_llm: bool = field(default=False)  # 使用LoRA时设为False

    model_max_length: int = field(
        default=2048,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = True  # 默认启用LoRA
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    protein_abstractor_lr: Optional[float] = None
    ligand_abstractor_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    save_safetensors: bool = False


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['protein_abstractor', 'ligand_abstractor',
                           'protein_encoder', 'ligand_encoder']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    ls = list(lora_module_names)
    print(f"LoRA target modules ({len(ls)}): {ls[:5]}...")
    return ls


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str,
                                  ):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict

        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make train/val/test datasets and collator for PLA supervised fine-tuning."""

    # 计算训练集的亲和力等级阈值和权重
    thresholds, level_names, weights = calculate_affinity_levels(
        data_args.data_path,
        num_levels=data_args.num_affinity_levels
    )

    rank0_print(f"Affinity level thresholds: {thresholds}")
    rank0_print(f"Level names: {level_names}")
    rank0_print(f"Interval median weights: {weights}")

    # 创建训练数据集
    train_dataset = PLADataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        data_args=data_args,
        thresholds=thresholds,
        level_names=level_names,
        weights=weights
    )

    # 创建验证数据集（如果提供）
    eval_dataset = None
    if data_args.val_data_path is not None:
        eval_dataset = PLADataset(
            data_path=data_args.val_data_path,
            tokenizer=tokenizer,
            data_args=data_args,
            thresholds=thresholds,  # 使用训练集的阈值
            level_names=level_names,
            weights=weights  # 使用训练集的权重
        )

    # 创建测试数据集（如果提供）
    test_dataset = None
    if data_args.test_data_path is not None:
        test_dataset = PLADataset(
            data_path=data_args.test_data_path,
            tokenizer=tokenizer,
            data_args=data_args,
            thresholds=thresholds,  # 使用训练集的阈值
            level_names=level_names,
            weights=weights  # 使用训练集的权重
        )

    # 创建数据整理器
    data_collator = PLADataCollator(tokenizer=tokenizer)

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
        data_collator=data_collator
    )


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            #device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    # 加载PLA模型
    from q_align.model.modeling_pla import PLALlamaForCausalLM
    from q_align.model.configuration_pla import PLAConfig
    from transformers import LlamaConfig, AutoModelForCausalLM

    # 1. 加载Llama2的基础配置
    rank0_print("Loading Llama2 base config...")
    llama_config = LlamaConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir
    )

    # 2. 创建PLA配置（包含pla_config字段）
    rank0_print("Creating PLA config...")
    pla_config = PLAConfig(**llama_config.to_dict())
    rank0_print(f"PLA config created with pla_config: {list(pla_config.pla_config.keys())}")

    # 3. 使用PLA配置初始化模型
    rank0_print("Initializing PLA model...")
    model = PLALlamaForCausalLM(pla_config)

    # 4. 直接从 safetensors 加载 LLaMA-2 预训练权重并复制到PLA模型
    rank0_print("Copying LLM weights to PLA model...")
    # replace_llama_modality_adaptive() 在 import 时替换了全局 LlamaDecoderLayer，
    # 导致 AutoModelForCausalLM.from_pretrained 构建的模型使用 MultiwayNetwork 结构，
    # from_pretrained 无法正确填充三路权重（三路均为随机初始化）。
    # 直接从 safetensors 文件读取原始 LLaMA-2 预训练权重，绕过模型构建。
    from safetensors import safe_open
    import json, re, os
    index_path = os.path.join(model_args.model_name_or_path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    shard_files = sorted(set(index["weight_map"].values()))
    raw_sd = {}
    for shard in shard_files:
        shard_path = os.path.join(model_args.model_name_or_path, shard)
        with safe_open(shard_path, framework="pt") as f:
            for k in f.keys():
                raw_sd[k] = f.get_tensor(k).to(compute_dtype)

    # key 格式: "model.layers.X.self_attn.k_proj.weight" -> 去掉 "model." 前缀
    # MultiwayNetwork 层需要展开到三路: "xxx.weight" -> "xxx.multiway.{0,1,2}.weight"
    multiway_suffixes = ["self_attn.k_proj", "self_attn.v_proj",
                         "input_layernorm", "post_attention_layernorm"]
    mapped_sd = {}
    lm_head_sd = {}
    for key, value in raw_sd.items():
        if key.startswith("model."):
            new_key = key[len("model."):]
        elif key.startswith("lm_head."):
            lm_head_sd[key[len("lm_head."):]] = value
            continue
        else:
            new_key = key
        expanded = False
        for mk in multiway_suffixes:
            if re.search(rf'\.{re.escape(mk)}\b', new_key):
                for i in range(3):
                    mapped_sd[re.sub(rf'\.{re.escape(mk)}\b', f'.{mk}.multiway.{i}', new_key)] = value.clone()
                expanded = True
                break
        if not expanded:
            mapped_sd[new_key] = value

    missing, unexpected = model.model.load_state_dict(mapped_sd, strict=False)
    rank0_print(f"Weight loading: {len(missing)} missing, {len(unexpected)} unexpected keys")
    if missing:
        rank0_print(f"Missing keys (first 10): {missing[:10]}")
    if lm_head_sd:
        model.lm_head.load_state_dict(lm_head_sd, strict=False)

    import gc
    gc.collect()

    rank0_print("PLA model initialized successfully")
    rank0_print(f"Model has protein_abstractor: {hasattr(model.model, 'protein_abstractor')}")
    rank0_print(f"Model has ligand_abstractor: {hasattr(model.model, 'ligand_abstractor')}")

    print(model.config)
    model.config.use_cache = False

    # 冻结策略
    if training_args.freeze_protein_encoder:
        rank0_print("Freezing protein encoder...")
        model.model.protein_encoder.requires_grad_(False)

    if training_args.freeze_ligand_encoder:
        rank0_print("Freezing ligand encoder...")
        model.model.ligand_encoder.requires_grad_(False)

    if not training_args.tune_protein_abstractor:
        rank0_print("Freezing protein abstractor...")
        model.model.protein_abstractor.requires_grad_(False)
    else:
        rank0_print("Training protein abstractor...")
        model.model.protein_abstractor.requires_grad_(True)

    if not training_args.tune_ligand_abstractor:
        rank0_print("Freezing ligand abstractor...")
        model.model.ligand_abstractor.requires_grad_(False)
    else:
        rank0_print("Training ligand abstractor...")
        model.model.ligand_abstractor.requires_grad_(True)

    if training_args.freeze_llm:
        rank0_print("Freezing LLM backbone...")
        for param in model.model.parameters():
            if param not in model.model.protein_abstractor.parameters() and \
               param not in model.model.ligand_abstractor.parameters():
                param.requires_grad = False

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")

        model = get_peft_model(model, lora_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.unk_token
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # 设置可训练参数（注意保留 abstractor 的 requires_grad）
    for n, p in model.named_parameters():
        if training_args.lora_enable:
            if "lora_" in n:
                p.requires_grad = True
            elif "protein_abstractor" in n and training_args.tune_protein_abstractor:
                p.requires_grad = True
            elif "ligand_abstractor" in n and training_args.tune_ligand_abstractor:
                p.requires_grad = True
            else:
                p.requires_grad = False
        else:
            p.requires_grad = True

    if training_args.lora_enable:
        model.print_trainable_parameters()

    # 诊断：验证各模块可训练参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    rank0_print(f"Trainable params: {trainable_params:,} / {all_params:,} = {100*trainable_params/all_params:.2f}%")
    for module_name in ["protein_abstractor", "ligand_abstractor", "lora_"]:
        count = sum(p.numel() for n, p in model.named_parameters() if module_name in n and p.requires_grad)
        rank0_print(f"  {module_name} trainable: {count:,}")

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)

    # PLATrainer does not take test_dataset in __init__.
    test_dataset = data_module.pop("test_dataset", None)

    # ----------------------------------------------------------------
    # 构建 PLATrainer 所需的 preferential_ids 和 weight_tensor
    # ----------------------------------------------------------------
    preferential_ids = get_preferential_ids(tokenizer)
    rank0_print(f"Preferential token IDs: {dict(zip(PLA_LEVEL_NAMES, preferential_ids))}")

    train_weights = data_module["train_dataset"].weights
    weight_tensor = torch.tensor(train_weights, dtype=torch.float32)
    rank0_print(f"Weight tensor (interval medians): {weight_tensor.tolist()}")

    train_thresholds = data_module["train_dataset"].thresholds

    trainer = PLATrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=make_compute_pla_metrics(train_thresholds),
        preferential_ids=preferential_ids,
        weight_tensor=weight_tensor,
        **data_module,
    )
    trainer.set_train_thresholds(train_thresholds)

    # if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
    #     trainer.train(resume_from_checkpoint=True)
    # else:
    #     trainer.train()
    
    # TODO I dont like auto resume << REMOVE IT AND UNCOMMENT THE ABOVE CODE
    trainer.train()

    trainer.save_state()

    # ----------------------------------------------------------------
    # 训练结束后在测试集上评估（使用验证集最优 checkpoint 的权重）
    # 若设置了 load_best_model_at_end=True（默认），trainer.train() 结束后
    # model 权重已自动还原为验证集最优 checkpoint。
    # ----------------------------------------------------------------
    if test_dataset is not None and \
            (training_args.local_rank == 0 or training_args.local_rank == -1):
        rank0_print("\n" + "=" * 60)
        rank0_print("Evaluating best checkpoint on test set ...")
        rank0_print(f"  Test CSV: {data_args.test_data_path}")
        if not training_args.load_best_model_at_end:
            rank0_print(
                "  WARNING: load_best_model_at_end=False, using final "
                "(not necessarily best) model weights."
            )

        test_output = trainer.predict(test_dataset)
        test_metrics = compute_affinity_metrics(
            test_output.predictions, test_output.label_ids, thresholds=train_thresholds
        )

        # Persist test metrics to output_dir for later comparison and reporting.
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)

        rank0_print("\n[Test Set Results]")
        rank0_print(f"  Samples  : {test_metrics['num_samples']}")
        rank0_print(f"  RMSE     : {test_metrics['rmse']:.4f}")
        rank0_print(f"  MAE      : {test_metrics['mae']:.4f}")
        rank0_print(f"  SD       : {test_metrics['sd']:.4f}")
        rank0_print(f"  Pearson R: {test_metrics['pearson_r']:.4f}  (p={test_metrics['pearson_p']:.2e})")
        rank0_print(f"  Spearman : {test_metrics['spearman_r']:.4f}  (p={test_metrics['spearman_p']:.2e})")
        rank0_print(f"  Accuracy : {test_metrics['accuracy']:.4f}")
        rank0_print("=" * 60 + "\n")

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir,
                                       )


if __name__ == "__main__":
    train()