import os
import torch
import numpy as np

from torch.utils.data import Sampler
from q_align.constants import IGNORE_INDEX

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional
from icecream import ic

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class MPLUGOwl2Trainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        #if is_sagemaker_mp_enabled():
        #    return super().create_optimizer()
        #if self.sharded_ddp == ShardedDDPOption.SIMPLE:
        #    return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            # 支持PLA模型的protein_abstractor和ligand_abstractor
            has_protein_abstractor_lr = hasattr(self.args, 'protein_abstractor_lr') and self.args.protein_abstractor_lr is not None
            has_ligand_abstractor_lr = hasattr(self.args, 'ligand_abstractor_lr') and self.args.ligand_abstractor_lr is not None
            has_visual_abstractor_lr = hasattr(self.args, 'visual_abstractor_lr') and self.args.visual_abstractor_lr is not None

            if has_protein_abstractor_lr or has_ligand_abstractor_lr or has_visual_abstractor_lr:
                # 收集需要特殊学习率的参数
                special_lr_parameters = []

                if has_protein_abstractor_lr:
                    protein_params = [name for name, _ in opt_model.named_parameters() if "protein_abstractor" in name]
                    special_lr_parameters.extend(protein_params)

                if has_ligand_abstractor_lr:
                    ligand_params = [name for name, _ in opt_model.named_parameters() if "ligand_abstractor" in name]
                    special_lr_parameters.extend(ligand_params)

                if has_visual_abstractor_lr:
                    visual_params = [name for name, _ in opt_model.named_parameters() if "visual_abstractor" in name]
                    special_lr_parameters.extend(visual_params)

                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

                # 为protein_abstractor添加参数组
                if has_protein_abstractor_lr:
                    optimizer_grouped_parameters.extend([
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n in decay_parameters and "protein_abstractor" in n and p.requires_grad)
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.protein_abstractor_lr,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n not in decay_parameters and "protein_abstractor" in n and p.requires_grad)
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.protein_abstractor_lr,
                        },
                    ])

                # 为ligand_abstractor添加参数组
                if has_ligand_abstractor_lr:
                    optimizer_grouped_parameters.extend([
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n in decay_parameters and "ligand_abstractor" in n and p.requires_grad)
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.ligand_abstractor_lr,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n not in decay_parameters and "ligand_abstractor" in n and p.requires_grad)
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.ligand_abstractor_lr,
                        },
                    ])

                # 为visual_abstractor添加参数组（兼容原始代码）
                if has_visual_abstractor_lr:
                    optimizer_grouped_parameters.extend([
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n in decay_parameters and "visual_abstractor" in n and p.requires_grad)
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.visual_abstractor_lr,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n not in decay_parameters and "visual_abstractor" in n and p.requires_grad)
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.visual_abstractor_lr,
                        },
                    ])
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            ic(len(optimizer_grouped_parameters[0]['params']),len(optimizer_grouped_parameters[1]['params']))
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if True:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            logger.info(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        super(MPLUGOwl2Trainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        super(MPLUGOwl2Trainer, self)._save(output_dir, state_dict)


class PLATrainer(MPLUGOwl2Trainer):
    """
    扩展 MPLUGOwl2Trainer，覆写 prediction_step 使验证时直接输出亲和力预测值，
    而不是完整词表 logits（内存友好），并支持 compute_metrics 计算 RMSE/MAE。
    """

    def __init__(self, *args, preferential_ids=None, weight_tensor=None, **kwargs):
        """
        Args:
            preferential_ids: List[int] — 5 个亲和力等级词在词表中的 token ID，
                              顺序为升序 pKd [negligible, weak, moderate, strong, potent]
            weight_tensor:    torch.FloatTensor of shape [5] — 每个等级对应的 pKd 中值，
                              顺序与 preferential_ids 一致
        """
        super().__init__(*args, **kwargs)
        self.preferential_ids = preferential_ids or []
        # 保持在 CPU；prediction_step 内会 .to(device) 使用
        self.weight_tensor = weight_tensor
        # 训练时累积预测值和真实值，用于 log 时计算 Pearson/Accuracy
        self._train_preds: List[float] = []
        self._train_labels: List[float] = []
        self._train_pred_classes: List[int] = []
        self._train_thresholds = None  # 由外部通过 set_train_thresholds 设置

    def _get_modality_expansion_offset(self, model) -> int:
        """
        计算多模态替换带来的序列净增长量：
        1 个 <protein> 占位符会被替换为 (protein_queries + 1) 个 token，净增 protein_queries；
        1 个 <ligand> 占位符会被替换为 (ligand_queries + 1) 个 token，净增 ligand_queries。
        因此总偏移量 = protein_queries + ligand_queries。
        """
        unwrapped_model = model.module if hasattr(model, "module") else model

        try:
            core_model = unwrapped_model.get_model()
            protein_queries = int(core_model.protein_abstractor.query_embeds.shape[1])
            ligand_queries = int(core_model.ligand_abstractor.query_embeds.shape[1])
            return protein_queries + ligand_queries
        except Exception:
            pass

        try:
            pla_cfg = unwrapped_model.config.pla_config
            protein_queries = int(pla_cfg["protein_abstractor"]["num_learnable_queries"])
            ligand_queries = int(pla_cfg["ligand_abstractor"]["num_learnable_queries"])
            return protein_queries + ligand_queries
        except Exception:
            return 0

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        验证时:
        1. 前向传播获得 loss 和 logits
        2. 取最后一个 token 位置的 5 个等级 logits → softmax → 加权求和 → 预测 pKd
        3. 返回 (loss, pred_affinities [batch], true_affinities [batch])
           而不是完整的 (loss, logits [batch, seq, vocab], labels [batch, seq])
        """
        has_labels = "labels" in inputs

        inputs = self._prepare_inputs(inputs)

        # 分布式评估下 gather 走 GPU backend，不能在这里提前转到 CPU。
        true_affinities = inputs.get("affinities", None)
        if isinstance(true_affinities, torch.Tensor):
            true_affinities = true_affinities.detach().float()

        with torch.no_grad():
            outputs = model(**inputs)

        loss = None
        if has_labels and outputs.loss is not None:
            loss = outputs.loss.mean().detach()

        if prediction_loss_only:
            return (loss, None, None)

        logits = outputs.logits  # [batch, seq_len, vocab_size]

        if logits is not None and len(self.preferential_ids) > 0:
            # inputs["labels"] 是原始未展开序列，logits 是模型内部插入蛋白/配体特征后的展开序列。
            # 这里优先定位“第一个等级词 token”在 labels 中的位置，
            # 再映射到展开序列并取其前一个位置的 logit（下一个 token 预测）。
            expansion_offset = self._get_modality_expansion_offset(model)

            labels_in = inputs.get("labels", None)
            if labels_in is not None:
                pref_ids_tensor = torch.tensor(self.preferential_ids, device=labels_in.device, dtype=labels_in.dtype)
                pref_token_mask = (labels_in.unsqueeze(-1) == pref_ids_tensor.view(1, 1, -1)).any(dim=-1)  # [batch, seq]
                has_pref_token = pref_token_mask.any(dim=1)  # [batch]

                first_pref_pos = pref_token_mask.int().argmax(dim=1)  # [batch]
                first_non_ignore_pos = (labels_in != IGNORE_INDEX).int().argmax(dim=1)  # [batch]
                target_pos = torch.where(has_pref_token, first_pref_pos, first_non_ignore_pos)  # [batch]

                logit_pos = (target_pos - 1 + expansion_offset).clamp(min=0, max=logits.shape[1] - 1)  # [batch]

                # 验证 labels 中 target_pos 确实是 preferential token
                pref_set = set(self.preferential_ids)
                for b in range(labels_in.shape[0]):
                    if has_pref_token[b]:
                        tok = labels_in[b, target_pos[b]].item()
                        assert tok in pref_set, (
                            f"Logit position mismatch: labels[{b}, {target_pos[b]}] = {tok}, "
                            f"expected one of {self.preferential_ids}"
                        )

                batch_idx = torch.arange(logits.shape[0], device=logits.device)
                aff_logits = logits[batch_idx, logit_pos][:, self.preferential_ids].float()  # [batch, 5]
            else:
                aff_logits = logits[:, -1, self.preferential_ids].float()  # fallback
            probs = torch.softmax(aff_logits, dim=-1)                   # [batch, 5]
            wt = self.weight_tensor.to(probs.device).float()            # [5]
            pred_affinities = (probs @ wt).detach()                     # [batch]
            pred_classes = probs.argmax(dim=-1).detach().float()         # [batch]
            # 打包: predictions shape [batch, 2] — 第0列=pred_pkd, 第1列=pred_class
            predictions = torch.stack([pred_affinities, pred_classes], dim=-1)  # [batch, 2]
        else:
            predictions = None

        return (loss, predictions, true_affinities)

    def set_train_thresholds(self, thresholds):
        """由 train_pla.py 在创建 trainer 后调用，传入训练集阈值。"""
        self._train_thresholds = thresholds

    def _extract_train_pred(self, model, inputs):
        """从当前 batch 的 logits 提取预测亲和力和真实亲和力，不计算梯度。"""
        if not self.preferential_ids or self.weight_tensor is None:
            return None, None, None
        true_affinities = inputs.get("affinities", None)
        if true_affinities is None:
            return None, None, None
        logits = model(**{k: v for k, v in inputs.items()}).logits
        expansion_offset = self._get_modality_expansion_offset(model)
        labels_in = inputs.get("labels", None)
        if labels_in is not None:
            pref_ids_tensor = torch.tensor(self.preferential_ids, device=labels_in.device, dtype=labels_in.dtype)
            pref_token_mask = (labels_in.unsqueeze(-1) == pref_ids_tensor.view(1, 1, -1)).any(dim=-1)
            has_pref_token = pref_token_mask.any(dim=1)
            first_pref_pos = pref_token_mask.int().argmax(dim=1)
            from q_align.constants import IGNORE_INDEX as _IGNORE
            first_non_ignore_pos = (labels_in != _IGNORE).int().argmax(dim=1)
            target_pos = torch.where(has_pref_token, first_pref_pos, first_non_ignore_pos)
            logit_pos = (target_pos - 1 + expansion_offset).clamp(min=0, max=logits.shape[1] - 1)
            batch_idx = torch.arange(logits.shape[0], device=logits.device)
            aff_logits = logits[batch_idx, logit_pos][:, self.preferential_ids].float()
        else:
            aff_logits = logits[:, -1, self.preferential_ids].float()
        probs = torch.softmax(aff_logits, dim=-1)
        wt = self.weight_tensor.to(probs.device).float()
        preds = (probs @ wt).detach().cpu().numpy()
        pred_classes = probs.argmax(dim=-1).detach().cpu().numpy()
        labels = true_affinities.detach().float().cpu().numpy()
        return preds, pred_classes, labels

    def training_step(self, model, inputs, num_items_in_batch=None):
        # 正常训练步
        if num_items_in_batch is not None:
            loss = super().training_step(model, inputs, num_items_in_batch)
        else:
            loss = super().training_step(model, inputs)

        # 每 50 步才做一次额外 forward 来计算训练指标，避免每步双重 forward 的开销
        if self.state.global_step % 50 == 0:
            try:
                with torch.no_grad():
                    preds, pred_classes, labels = self._extract_train_pred(model, self._prepare_inputs(inputs))
                if preds is not None:
                    self._train_preds.extend(preds.tolist())
                    self._train_labels.extend(labels.tolist())
                    self._train_pred_classes.extend(pred_classes.tolist())
            except Exception:
                pass

        return loss

    def log(self, logs, start_time=None):
        if "loss" in logs and len(self._train_preds) > 1:
            preds = np.array(self._train_preds, dtype=np.float64)
            lbls = np.array(self._train_labels, dtype=np.float64)
            valid = np.isfinite(preds) & np.isfinite(lbls)
            preds, lbls = preds[valid], lbls[valid]
            if len(preds) > 1:
                from scipy.stats import pearsonr
                pearson_r, _ = pearsonr(lbls, preds)
                logs["train_pearson_r"] = round(float(pearson_r), 4)
                if self._train_thresholds is not None and len(self._train_pred_classes) > 0:
                    pred_cls = np.array(self._train_pred_classes, dtype=np.int64)
                    pred_cls = pred_cls[valid] if len(pred_cls) == len(valid) else pred_cls[:len(lbls)]
                    # 真实类别从连续值映射
                    thresholds = self._train_thresholds
                    def _to_cls(vals):
                        cls = np.zeros(len(vals), dtype=np.int64)
                        for i in range(len(thresholds) - 1):
                            if i == len(thresholds) - 2:
                                mask = vals >= thresholds[i]
                            else:
                                mask = (vals >= thresholds[i]) & (vals < thresholds[i + 1])
                            cls[mask] = i
                        return cls
                    true_cls = _to_cls(lbls)
                    acc = float(np.mean(pred_cls[:len(true_cls)] == true_cls))
                    logs["train_accuracy"] = round(acc, 4)
            # 清空累积缓冲
            self._train_preds.clear()
            self._train_labels.clear()
            self._train_pred_classes.clear()

        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)