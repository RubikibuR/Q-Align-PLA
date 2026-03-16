# Copyright (c) 2024 PLA Project
# Configuration for Protein-Ligand Affinity Prediction Model

import copy
import os
from typing import Union

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from .configuration_mplug_owl2 import LlamaConfig, MplugOwlVisualAbstractorConfig

logger = logging.get_logger(__name__)


class ProteinEncoderConfig(PretrainedConfig):
    """
    蛋白质编码器配置类
    """
    model_type = "protein_encoder"

    def __init__(
        self,
        model_name="facebook/esm2_t33_650M_UR50D",
        hidden_size=1280,
        max_length=1024,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.max_length = max_length


class LigandEncoderConfig(PretrainedConfig):
    """
    配体编码器配置类
    """
    model_type = "ligand_encoder"

    def __init__(
        self,
        model_name="DeepChem/ChemBERTa-77M-MTR",
        hidden_size=384,  # ChemBERTa-77M-MTR的实际hidden_size
        max_length=512,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.max_length = max_length


class ProteinAbstractorConfig(PretrainedConfig):
    """
    蛋白质抽象器配置类
    """
    model_type = "protein_abstractor"

    def __init__(
        self,
        num_learnable_queries=64,
        num_hidden_layers=6,
        hidden_size=1280,
        encoder_hidden_size=1280,  # 蛋白质编码器的输出维度
        num_attention_heads=16,
        intermediate_size=5120,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_learnable_queries = num_learnable_queries
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.encoder_hidden_size = encoder_hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps


class LigandAbstractorConfig(PretrainedConfig):
    """
    配体抽象器配置类
    """
    model_type = "ligand_abstractor"

    def __init__(
        self,
        num_learnable_queries=32,
        num_hidden_layers=6,
        hidden_size=384,
        encoder_hidden_size=384,  # 配体编码器的输出维度（ChemBERTa-77M-MTR）
        num_attention_heads=12,
        intermediate_size=3072,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_learnable_queries = num_learnable_queries
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.encoder_hidden_size = encoder_hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps


DEFAULT_PLA_CONFIG = {
    "protein_encoder": ProteinEncoderConfig().to_dict(),
    "ligand_encoder": LigandEncoderConfig().to_dict(),
    "protein_abstractor": ProteinAbstractorConfig().to_dict(),
    "ligand_abstractor": LigandAbstractorConfig().to_dict()
}


class PLAConfig(LlamaConfig):
    """
    PLA模型配置类，继承自LlamaConfig
    整合蛋白质编码器、配体编码器、蛋白质抽象器、配体抽象器和LLM的配置
    """
    model_type = "pla_model"

    def __init__(
        self,
        pla_config=None,
        use_cache=True,
        **kwargs
    ):
        if pla_config is None:
            self.pla_config = DEFAULT_PLA_CONFIG
        else:
            self.pla_config = pla_config

        super().__init__(
            use_cache=use_cache,
            **kwargs,
        )


if __name__ == "__main__":
    print("Protein Encoder Config:")
    print(ProteinEncoderConfig().to_dict())
    print("\nLigand Encoder Config:")
    print(LigandEncoderConfig().to_dict())
    print("\nProtein Abstractor Config:")
    print(ProteinAbstractorConfig().to_dict())
    print("\nLigand Abstractor Config:")
    print(LigandAbstractorConfig().to_dict())
    print("\nPLA Config:")
    print(PLAConfig().to_dict())
