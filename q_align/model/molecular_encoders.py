import torch
import torch.nn as nn
from transformers import EsmModel, EsmTokenizer, RobertaModel, RobertaTokenizer


class ProteinEncoder(nn.Module):
    """
    蛋白质编码器，基于ESM-2预训练模型
    输入: 蛋白质序列（氨基酸字符串列表）
    输出: 蛋白质表示 (batch, seq_len, hidden_dim=1280)
    """
    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D", max_length=1024):
        super().__init__()
        self.model = EsmModel.from_pretrained(model_name)
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.max_length = max_length

        # 冻结参数
        for param in self.model.parameters():
            param.requires_grad = False

        self.hidden_size = self.model.config.hidden_size  # 1280 for esm2_t33_650M_UR50D

    def forward(self, protein_sequences):
        """
        Args:
            protein_sequences: List[str], 蛋白质序列列表
        Returns:
            tuple: (hidden_states, attention_mask)
                - hidden_states: (batch, seq_len, 1280)
                - attention_mask: (batch, seq_len)
        """
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            protein_sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.last_hidden_state, inputs['attention_mask']


class LigandEncoder(nn.Module):
    """
    配体编码器，基于ChemBERTa预训练模型
    输入: SMILES字符串列表
    输出: 配体表示 (batch, seq_len, hidden_dim=384)
    """
    def __init__(self, model_name="DeepChem/ChemBERTa-77M-MTR", max_length=512):
        super().__init__()
        self.model = RobertaModel.from_pretrained(model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.max_length = max_length

        # 冻结参数
        for param in self.model.parameters():
            param.requires_grad = False

        self.hidden_size = self.model.config.hidden_size  # 384 for ChemBERTa

    def forward(self, smiles_list):
        """
        Args:
            smiles_list: List[str], SMILES字符串列表
        Returns:
            tuple: (hidden_states, attention_mask)
                - hidden_states: (batch, seq_len, 384)
                - attention_mask: (batch, seq_len)
        """
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            smiles_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.last_hidden_state, inputs['attention_mask']
