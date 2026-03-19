import os
import copy
import json
import pandas as pd
import numpy as np
import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

import torch
from torch.utils.data import Dataset

from q_align.constants import IGNORE_INDEX, PROTEIN_TOKEN_INDEX, LIGAND_TOKEN_INDEX, DEFAULT_PROTEIN_TOKEN, DEFAULT_LIGAND_TOKEN, PLA_LEVEL_NAMES
from q_align import conversation as conversation_lib
from q_align.mm_utils import tokenizer_multimodal_token

from icecream import ic


def calculate_affinity_levels(train_csv_path, num_levels=5):
    """
    根据训练集的亲和力分布计算等级划分阈值（使用分位数）
    同时计算每个区间的中点作为权重
    Args:
        train_csv_path: 训练集CSV路径
        num_levels: 等级数量
    Returns:
        thresholds: 阈值列表
        level_names: 等级名称列表
        weights: 每个等级的权重（区间中点）
    """
    df = pd.read_csv(train_csv_path)
    affinities = df['Y'].values

    # 计算分位数作为阈值
    quantiles = np.linspace(0, 1, num_levels + 1)
    thresholds = np.quantile(affinities, quantiles)

    # 计算每个区间的中点作为权重
    weights = []
    for i in range(num_levels):
        weights.append(float((thresholds[i] + thresholds[i + 1]) / 2))

    # 定义等级名称（从高到低）
    # 注意：必须使用单 token 词，否则推理时 id_[1] 提取会出错
    # （"very strong"/"very weak" 都会提取到相同的 "very" token）
    if num_levels == 5:
        level_names = list(PLA_LEVEL_NAMES)
    elif num_levels == 3:
        level_names = list(PLA_LEVEL_NAMES[1:4])
    else:
        level_names = [f"level_{i}" for i in range(num_levels)]

    return thresholds, level_names, weights


def affinity_to_level(affinity, thresholds, level_names):
    """
    将亲和力值映射到等级
    """
    for i in range(len(thresholds) - 1):
        if thresholds[i] <= affinity < thresholds[i + 1]:
            return level_names[i]
    # 处理边界情况
    if affinity >= thresholds[-1]:
        return level_names[-1]
    return level_names[0]


def preprocess_pla(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_protein_ligand: bool = True
) -> Dict:
    """
    预处理PLA数据，将对话转换为input_ids和labels
    """
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # 应用prompt模板
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # 跳过第一轮不是human的对话
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize对话
    if has_protein_ligand:
        input_ids = torch.stack([tokenizer_multimodal_token(prompt, tokenizer, return_tensors='pt')
                                for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # 屏蔽掉非assistant部分的标签
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_protein_ligand:
                round_len = len(tokenizer_multimodal_token(rou, tokenizer))
                instruction_len = len(tokenizer_multimodal_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


class PLADataset(Dataset):
    """
    PLA数据集类，用于加载和处理蛋白-配体亲和力预测数据
    """
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer,
                 data_args, thresholds=None, level_names=None, weights=None):
        super(PLADataset, self).__init__()

        # 读取CSV数据
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.data_args = data_args

        print(f"Loaded {len(self.data)} samples from {data_path}")

        # 如果没有提供阈值，则根据当前数据计算
        if thresholds is None or level_names is None or weights is None:
            self.thresholds, self.level_names, self.weights = calculate_affinity_levels(data_path, num_levels=5)
            print(f"Affinity thresholds: {self.thresholds}")
            print(f"Level names: {self.level_names}")
            print(f"Interval weights: {self.weights}")
        else:
            self.thresholds = thresholds
            self.level_names = level_names
            self.weights = weights

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[i]

        smiles = row['SMILES']
        protein_seq = row['Protein']
        affinity = row['Y']

        # 将亲和力映射到等级
        affinity_level = affinity_to_level(affinity, self.thresholds, self.level_names)

        # 构建对话格式（使用简洁风格的prompt）
        conversations = [
            {
                "from": "human",
                "value": f"Predict the binding affinity rating between this protein and ligand.\n{DEFAULT_PROTEIN_TOKEN}\n{DEFAULT_LIGAND_TOKEN}"
            },
            {
                "from": "gpt",
                "value": f"{affinity_level}"
            }
        ]

        # 预处理数据
        data_dict = preprocess_pla(
            [conversations],
            self.tokenizer,
            has_protein_ligand=True
        )

        # 返回数据
        data_dict = dict(
            input_ids=data_dict["input_ids"][0],
            labels=data_dict["labels"][0],
            protein_sequence=protein_seq,
            smiles=smiles,
            affinity=affinity
        )

        return data_dict


@dataclass
class PLADataCollator:
    """
    PLA数据整理器，用于批处理
    """
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance['input_ids'] for instance in instances]
        labels = [instance['labels'] for instance in instances]
        protein_sequences = [instance['protein_sequence'] for instance in instances]
        smiles_list = [instance['smiles'] for instance in instances]
        affinities = [instance['affinity'] for instance in instances]

        # Padding
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX
        )

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            protein_sequences=protein_sequences,
            smiles_list=smiles_list,
            affinities=torch.tensor(affinities, dtype=torch.float32),
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        return batch
