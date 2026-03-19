"""
PLA Inference Script
加载指定路径的模型，对给定蛋白质序列和配体SMILES进行推理，
输出文本回答以及预测的亲和力评分（pKd）。

Usage:
    python infer_pla.py \
        --model_path /path/to/checkpoint \
        --base_model_path /path/to/llama2 \
        --protein "MKTAYIAKQRQISFVKSHFSRQ..." \
        --smiles "CC1=CC=C(C=C1)NC(=O)..." \
        [--train_csv /path/to/train.csv]  # 用于恢复权重向量，可选

    # 也可以从CSV文件批量推理：
    python infer_pla.py \
        --model_path /path/to/checkpoint \
        --base_model_path /path/to/llama2 \
        --input_csv /path/to/input.csv \
        [--train_csv /path/to/train.csv]
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import transformers
from peft import PeftModel

from q_align import conversation as conversation_lib
from q_align.constants import (
    DEFAULT_LIGAND_TOKEN,
    DEFAULT_PROTEIN_TOKEN,
    IGNORE_INDEX,
    PLA_LEVEL_NAMES,
)
from q_align.mm_utils import tokenizer_multimodal_token
from q_align.model.configuration_pla import PLAConfig
from q_align.model.modeling_pla import PLALlamaForCausalLM
from q_align.train.pla_dataset import calculate_affinity_levels


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------

def load_model(model_path: str, base_model_path: str, device: str = "cuda"):
    """
    加载 PLALlamaForCausalLM + LoRA adapter。

    Args:
        model_path:      训练输出目录（含 adapter_config.json / non_lora_trainables.bin）
        base_model_path: LLaMA-2 基础模型路径
        device:          推理设备
    Returns:
        (model, tokenizer)
    """
    print(f"Loading base config from {base_model_path} ...")
    from transformers import LlamaConfig

    llama_config = LlamaConfig.from_pretrained(base_model_path)
    pla_config = PLAConfig(**llama_config.to_dict())

    print("Initializing PLA model ...")
    model = PLALlamaForCausalLM(pla_config)

    print("Loading LLaMA-2 pretrained weights ...")
    # replace_llama_modality_adaptive() 在 import 时替换了全局 LlamaDecoderLayer，
    # 导致 AutoModelForCausalLM.from_pretrained 构建的模型使用 MultiwayNetwork 结构，
    # from_pretrained 无法正确填充三路权重（三路均为随机初始化）。
    # 直接从 safetensors 文件读取原始 LLaMA-2 预训练权重，绕过模型构建。
    import re, json
    from safetensors import safe_open
    index_path = os.path.join(base_model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    shard_files = sorted(set(index["weight_map"].values()))
    raw_sd = {}
    for shard in shard_files:
        with safe_open(os.path.join(base_model_path, shard), framework="pt") as f:
            for k in f.keys():
                raw_sd[k] = f.get_tensor(k).to(torch.bfloat16)

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
    print(f"Weight loading: {len(missing)} missing, {len(unexpected)} unexpected keys")
    if lm_head_sd:
        model.lm_head.load_state_dict(lm_head_sd, strict=False)
    import gc; gc.collect()

    # 加载 non-LoRA 可训练参数（abstractor 权重）
    non_lora_path = os.path.join(model_path, "non_lora_trainables.bin")
    if os.path.exists(non_lora_path):
        print(f"Loading non-LoRA trainables from {non_lora_path} ...")
        non_lora_state = torch.load(non_lora_path, map_location="cpu")
        # 去掉 "base_model.model." 前缀（peft 保存时会加）
        cleaned = {}
        for k, v in non_lora_state.items():
            new_k = k.replace("base_model.model.", "")
            cleaned[new_k] = v
        model.load_state_dict(cleaned, strict=False)
    else:
        print("WARNING: non_lora_trainables.bin not found, abstractor weights may be random.")

    # 加载 LoRA adapter
    adapter_cfg = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        print(f"Loading LoRA adapter from {model_path} ...")
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
        print("LoRA merged.")
    else:
        print("No adapter_config.json found, skipping LoRA loading.")

    model = model.to(device=device, dtype=torch.bfloat16)
    model.config.use_cache = True
    model.eval()

    print("Loading tokenizer ...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model_path,
        model_max_length=2048,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    # 使用 vicuna_v1 对话模板（与训练一致）
    conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    return model, tokenizer


# ---------------------------------------------------------------------------
# 权重向量（pKd 区间中点）
# ---------------------------------------------------------------------------

def get_weight_tensor(train_csv: str, num_levels: int = 5) -> torch.Tensor:
    """
    从训练集 CSV 恢复权重向量；若未提供则使用默认值。
    """
    if train_csv and os.path.exists(train_csv):
        _, _, weights = calculate_affinity_levels(train_csv, num_levels=num_levels)
        print(f"Weight tensor from train CSV: {weights}")
        return torch.tensor(weights, dtype=torch.float32)
    # 默认权重（训练集未知时的合理估计）
    default_weights = [3.0, 5.0, 6.5, 7.5, 9.0]
    print(f"WARNING: train_csv not provided, using default weights: {default_weights}")
    return torch.tensor(default_weights, dtype=torch.float32)


def get_preferential_ids(tokenizer) -> list[int]:
    return [ids[1] for ids in tokenizer(PLA_LEVEL_NAMES)["input_ids"]]


# ---------------------------------------------------------------------------
# 单样本推理
# ---------------------------------------------------------------------------

def build_input(protein_seq: str, smiles: str, tokenizer) -> dict:
    """构建单样本的 model input（batch_size=1）。"""
    conv = conversation_lib.default_conversation.copy()
    conv.append_message(conv.roles[0],
        f"Predict the binding affinity rating between this protein and ligand.\n"
        f"{DEFAULT_PROTEIN_TOKEN}\n{DEFAULT_LIGAND_TOKEN}")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_multimodal_token(prompt, tokenizer, return_tensors="pt")  # [seq]
    input_ids = input_ids.unsqueeze(0)  # [1, seq]

    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)  # all tokens valid

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "protein_sequences": [protein_seq],
        "smiles_list": [smiles],
    }


@torch.no_grad()
def infer_single(
    model,
    tokenizer,
    protein_seq: str,
    smiles: str,
    preferential_ids: list[int],
    weight_tensor: torch.Tensor,
    device: str = "cuda",
    max_new_tokens: int = 32,
) -> dict:
    """
    对单个蛋白-配体对进行推理。
    Returns:
        {
            "answer":     str,    # 模型生成的文本回答
            "level":      str,    # 预测的亲和力等级名称
            "pred_pkd":   float,  # 预测的 pKd 值
            "probs":      list,   # 5 个等级的概率
        }
    """
    inputs = build_input(protein_seq, smiles, tokenizer)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    input_ids = inputs["input_ids"]  # [1, seq]
    protein_sequences = inputs["protein_sequences"]
    smiles_list = inputs["smiles_list"]

    # ---- 第一次 forward ----
    outputs = model(
        input_ids=input_ids,
        attention_mask=torch.ones(input_ids.shape, dtype=torch.long, device=device),
        protein_sequences=protein_sequences,
        smiles_list=smiles_list,
    )
    logits = outputs.logits  # [1, expanded_seq, vocab]

    next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
    generated_ids = [next_token_id.item()]
    eos_token_id = tokenizer.eos_token_id
    cur_ids = torch.cat([input_ids, next_token_id], dim=1)  # [1, seq+1]

    # ---- 自回归生成文本回答（无 KV cache，每步完整 forward）----
    for _ in range(max_new_tokens - 1):
        attn_mask = torch.ones(cur_ids.shape, dtype=torch.long, device=device)
        out = model(
            input_ids=cur_ids,
            attention_mask=attn_mask,
            protein_sequences=protein_sequences,
            smiles_list=smiles_list,
        )
        next_token_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        token_id = next_token_id.item()
        if token_id == eos_token_id:
            break
        if token_id in preferential_ids:
            aff_logits = out.logits[0, -1, preferential_ids].float()  
            probs = torch.softmax(aff_logits, dim=-1)                
            wt = weight_tensor.to(probs.device).float()
            pred_pkd = float((probs @ wt).item())
            level_idx = int(probs.argmax().item())
            level_name = PLA_LEVEL_NAMES[level_idx]

        generated_ids.append(token_id)
        cur_ids = torch.cat([cur_ids, next_token_id], dim=1)

    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return {
        "answer": answer,
        "level": level_name,
        "pred_pkd": pred_pkd,
        "probs": probs.cpu().tolist(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="PLA Inference")
    parser.add_argument("--model_path", default="pla-align-debug/baseline-pdbbind",
                        help="训练输出目录（含 LoRA adapter 和 non_lora_trainables.bin）")
    parser.add_argument("--base_model_path", default="/data/chenruixi/resources/modelscope/llama2-7b",
                        help="LLaMA-2 基础模型路径")
    parser.add_argument("--train_csv", default="datasets/pdbbind/random/train.csv",
                        help="训练集 CSV 路径，用于恢复权重向量（可选）")
    parser.add_argument("--device", default="cuda")

    # 单样本模式
    parser.add_argument("--protein", default=None, help="蛋白质氨基酸序列")
    parser.add_argument("--smiles", default=None, help="配体 SMILES 字符串")

    # 批量模式
    parser.add_argument("--input_csv", default="datasets/pdbbind/random/test.csv",
                        help="批量推理 CSV（需含 Protein / SMILES 列，可选含 Y 列）")
    parser.add_argument("--output_csv", default="./results.csv",
                        help="批量推理结果保存路径（默认打印到 stdout）")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.protein is None and args.input_csv is None:
        print("ERROR: 请提供 --protein + --smiles 或 --input_csv", file=sys.stderr)
        sys.exit(1)

    model, tokenizer = load_model(args.model_path, args.base_model_path, args.device)
    preferential_ids = get_preferential_ids(tokenizer)
    weight_tensor = get_weight_tensor(args.train_csv)

    print(f"Preferential IDs: {dict(zip(PLA_LEVEL_NAMES, preferential_ids))}")
    print(f"Weight tensor:    {weight_tensor.tolist()}")

    # ---- 单样本 ----
    if args.protein is not None:
        if args.smiles is None:
            print("ERROR: --smiles 是必须的", file=sys.stderr)
            sys.exit(1)
        result = infer_single(
            model, tokenizer,
            args.protein, args.smiles,
            preferential_ids, weight_tensor,
            device=args.device,
        )
        print("\n=== Inference Result ===")
        print(f"Answer  : {result['answer']}")
        print(f"Level   : {result['level']}")
        print(f"Pred pKd: {result['pred_pkd']:.4f}")
        print(f"Probs   : {dict(zip(PLA_LEVEL_NAMES, [f'{p:.4f}' for p in result['probs']]))}")
        return

    # ---- 批量 ----
    df = pd.read_csv(args.input_csv)
    assert "Protein" in df.columns and "SMILES" in df.columns, \
        "input_csv 必须包含 Protein 和 SMILES 列"

    results = []
    for i, row in df.iterrows():
        print(f"[{i+1}/{len(df)}] inferring ...", end="\r")
        res = infer_single(
            model, tokenizer,
            row["Protein"], row["SMILES"],
            preferential_ids, weight_tensor,
            device=args.device,
        )
        results.append(res)

    print()
    df["pred_pkd"] = [r["pred_pkd"] for r in results]
    df["pred_level"] = [r["level"] for r in results]
    df["answer"] = [r["answer"] for r in results]

    if "Y" in df.columns:
        from scipy.stats import pearsonr
        from sklearn.metrics import mean_squared_error
        from q_align.train.pla_dataset import affinity_to_level
        preds = df["pred_pkd"].values
        labels = df["Y"].values
        rmse = float(np.sqrt(mean_squared_error(labels, preds)))
        r, _ = pearsonr(labels, preds)

        # 分类准确率：将真实 pKd 转换为等级，与预测等级比较
        thresholds, level_names, _ = calculate_affinity_levels(args.train_csv)
        true_levels = [affinity_to_level(y, thresholds, level_names) for y in labels]
        df["true_level"] = true_levels
        correct = sum(t == p for t, p in zip(true_levels, df["pred_level"].values))
        acc = correct / len(true_levels)

        print(f"\nBatch metrics — RMSE: {rmse:.4f}  Pearson R: {r:.4f}  Acc: {acc:.4f}")

    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"Results saved to {args.output_csv}")
    else:
        print(df[["SMILES", "Protein", "pred_pkd", "pred_level", "answer"]].to_string())


if __name__ == "__main__":
    main()
