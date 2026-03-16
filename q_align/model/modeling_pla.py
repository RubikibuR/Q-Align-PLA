#    Copyright 2024 PLA Project
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

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import copy
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_pla import PLAConfig, ProteinEncoderConfig, LigandEncoderConfig, ProteinAbstractorConfig, LigandAbstractorConfig
from .molecular_encoders import ProteinEncoder, LigandEncoder
from .visual_encoder import ProteinAbstractorModel, LigandAbstractorModel
from .modeling_llama2 import replace_llama_modality_adaptive

from q_align.constants import IGNORE_INDEX, PROTEIN_TOKEN_INDEX, LIGAND_TOKEN_INDEX

from icecream import ic


class PLAMetaModel:
    """
    PLA元模型类，负责初始化蛋白质编码器、配体编码器和抽象器
    """
    def __init__(self, config):
        super(PLAMetaModel, self).__init__(config)

        # 初始化蛋白质编码器
        protein_encoder_config = ProteinEncoderConfig(**config.pla_config["protein_encoder"])
        self.protein_encoder = ProteinEncoder(
            model_name=protein_encoder_config.model_name,
            max_length=protein_encoder_config.max_length
        )

        # 初始化配体编码器
        ligand_encoder_config = LigandEncoderConfig(**config.pla_config["ligand_encoder"])
        self.ligand_encoder = LigandEncoder(
            model_name=ligand_encoder_config.model_name,
            max_length=ligand_encoder_config.max_length
        )

        # 初始化蛋白质抽象器
        protein_abstractor_config = ProteinAbstractorConfig(**config.pla_config["protein_abstractor"])
        self.protein_abstractor = ProteinAbstractorModel(
            protein_abstractor_config,
            config.hidden_size
        )

        # 初始化配体抽象器
        ligand_abstractor_config = LigandAbstractorConfig(**config.pla_config["ligand_abstractor"])
        self.ligand_abstractor = LigandAbstractorModel(
            ligand_abstractor_config,
            config.hidden_size
        )

    def encode_protein_ligand(self, protein_sequences, smiles_list):
        """
        编码蛋白质和配体
        Args:
            protein_sequences: List[str], 蛋白质序列列表
            smiles_list: List[str], SMILES字符串列表
        Returns:
            protein_features: (batch, num_protein_queries+1, hidden_size)
            ligand_features: (batch, num_ligand_queries+1, hidden_size)
        """
        # 编码蛋白质
        protein_hidden_states, protein_attention_mask = self.protein_encoder(protein_sequences)
        protein_features = self.protein_abstractor(
            encoder_hidden_states=protein_hidden_states,
            encoder_attention_mask=protein_attention_mask
        ).last_hidden_state

        # 编码配体
        ligand_hidden_states, ligand_attention_mask = self.ligand_encoder(smiles_list)
        ligand_features = self.ligand_abstractor(
            encoder_hidden_states=ligand_hidden_states,
            encoder_attention_mask=ligand_attention_mask
        ).last_hidden_state

        return protein_features, ligand_features


class PLAMetaForCausalLM(ABC):
    """
    PLA因果语言模型元类
    """
    @abstractmethod
    def get_model(self):
        pass

    def encode_protein_ligand(self, protein_sequences, smiles_list):
        return self.get_model().encode_protein_ligand(protein_sequences, smiles_list)

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels,
        protein_sequences=None, smiles_list=None
    ):
        """
        准备多模态输入：将蛋白质和配体特征插入到文本序列中
        """
        if protein_sequences is None or smiles_list is None:
            # 如果没有蛋白质或配体数据，直接返回
            if past_key_values is not None and labels is None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                                          dtype=attention_mask.dtype, device=attention_mask.device)
            modality_indicators = torch.zeros_like(input_ids).long()
            return input_ids, modality_indicators, attention_mask, past_key_values, None, labels

        # 编码蛋白质和配体
        protein_features, ligand_features = self.encode_protein_ligand(protein_sequences, smiles_list)

        new_input_embeds = []
        new_modality_indicators = []
        new_labels = [] if labels is not None else None

        for batch_idx, cur_input_ids in enumerate(input_ids):
            # 找到蛋白质和配体token的位置
            protein_token_indices = torch.where(cur_input_ids == PROTEIN_TOKEN_INDEX)[0]
            ligand_token_indices = torch.where(cur_input_ids == LIGAND_TOKEN_INDEX)[0]

            cur_new_input_embeds = []
            cur_modality_indicators = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []

            # 处理序列：按照protein和ligand token的位置分割
            # 假设顺序是: text -> <protein> -> text -> <ligand> -> text

            # 获取所有特殊token的位置并排序
            special_tokens = []
            for idx in protein_token_indices:
                special_tokens.append((idx.item(), 'protein'))
            for idx in ligand_token_indices:
                special_tokens.append((idx.item(), 'ligand'))
            special_tokens.sort(key=lambda x: x[0])

            cur_pos = 0
            for token_idx, token_type in special_tokens:
                # 添加token之前的文本
                if cur_pos < token_idx:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[cur_pos:token_idx]))
                    cur_modality_indicators.append(torch.zeros(token_idx - cur_pos).long())
                    if labels is not None:
                        cur_new_labels.append(cur_labels[cur_pos:token_idx])

                # 添加蛋白质或配体特征
                if token_type == 'protein':
                    cur_new_input_embeds.append(protein_features[batch_idx])
                    cur_modality_indicators.append(torch.ones(protein_features.shape[1]).long() * 1)  # 1表示蛋白质
                    if labels is not None:
                        cur_new_labels.append(torch.full((protein_features.shape[1],), IGNORE_INDEX,
                                                        device=labels.device, dtype=labels.dtype))
                elif token_type == 'ligand':
                    cur_new_input_embeds.append(ligand_features[batch_idx])
                    cur_modality_indicators.append(torch.ones(ligand_features.shape[1]).long() * 2)  # 2表示配体
                    if labels is not None:
                        cur_new_labels.append(torch.full((ligand_features.shape[1],), IGNORE_INDEX,
                                                        device=labels.device, dtype=labels.dtype))

                cur_pos = token_idx + 1

            # 添加最后的文本部分
            if cur_pos < len(cur_input_ids):
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[cur_pos:]))
                cur_modality_indicators.append(torch.zeros(len(cur_input_ids) - cur_pos).long())
                if labels is not None:
                    cur_new_labels.append(cur_labels[cur_pos:])

            # 拼接所有部分
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)

            cur_modality_indicators = [x.to(device=self.device) for x in cur_modality_indicators]
            cur_modality_indicators = torch.cat(cur_modality_indicators, dim=0)
            new_modality_indicators.append(cur_modality_indicators)

            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        # Padding到相同长度
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            # Padding embeddings
            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                                                                      dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            # Padding modality indicators
            new_modality_indicators_align = []
            for cur_modality_indicator in new_modality_indicators:
                cur_new_embed = torch.cat((cur_modality_indicator, torch.zeros(max_len - cur_modality_indicator.shape[0],
                                                                               dtype=cur_modality_indicator.dtype, device=cur_modality_indicator.device)), dim=0)
                new_modality_indicators_align.append(cur_new_embed)
            new_modality_indicators = torch.stack(new_modality_indicators_align, dim=0)

            # Padding labels
            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX,
                                                                         dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            # Padding attention mask
            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True,
                                                        dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False,
                                                         dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            new_modality_indicators = torch.stack(new_modality_indicators, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True,
                                                    dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, new_modality_indicators, attention_mask, past_key_values, new_input_embeds, new_labels


class PLALlamaModel(PLAMetaModel, LlamaModel):
    config_class = PLAConfig

    def __init__(self, config: PLAConfig):
        super(PLALlamaModel, self).__init__(config)


class PLALlamaForCausalLM(LlamaForCausalLM, PLAMetaForCausalLM):
    config_class = PLAConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = PLALlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        protein_sequences: Optional[List[str]] = None,
        smiles_list: Optional[List[str]] = None,
        affinities: Optional[torch.FloatTensor] = None,  # 添加亲和力参数
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 准备多模态输入
        input_ids, modality_indicators, attention_mask, past_key_values, inputs_embeds, labels = \
            self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels,
                                                      protein_sequences, smiles_list)

        # 前向传播
        outputs = self.model(
            input_ids=input_ids,
            modality_indicators=modality_indicators,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "protein_sequences": kwargs.get("protein_sequences", None),
                "smiles_list": kwargs.get("smiles_list", None),
            }
        )
        return model_inputs


AutoConfig.register("pla_model", PLAConfig)
AutoModelForCausalLM.register(PLAConfig, PLALlamaForCausalLM)
