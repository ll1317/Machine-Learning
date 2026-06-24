from __future__ import annotations

import torch
from torch import nn


class TransformerForecaster(nn.Module):
    """
    多变量 Transformer Encoder 直接多步预测模型。

    输入:
        x: [batch_size, input_length, input_dim]

    输出:
        y_hat: [batch_size, horizon]
    """

    def __init__(
        self,
        input_dim: int,
        input_length: int,
        horizon: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim 必须大于0。")
        if input_length <= 0:
            raise ValueError("input_length 必须大于0。")
        if horizon <= 0:
            raise ValueError("horizon 必须大于0。")
        if d_model <= 0:
            raise ValueError("d_model 必须大于0。")
        if d_model % nhead != 0:
            raise ValueError("d_model 必须能够被 nhead 整除。")

        self.input_dim = input_dim
        self.input_length = input_length
        self.horizon = horizon
        self.d_model = d_model

        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, d_model)

        # 可学习位置编码，长度与历史窗口90天一致。
        self.position_embedding = nn.Parameter(
            torch.zeros(1, input_length, d_model)
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # 对90天编码结果做平均池化，再一次性输出未来H天。
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"输入应为 [B,T,F] 三维张量，实际形状为 {tuple(x.shape)}"
            )
        if x.shape[1] != self.input_length:
            raise ValueError(
                f"模型要求输入长度 {self.input_length}，实际为 {x.shape[1]}。"
            )
        if x.shape[2] != self.input_dim:
            raise ValueError(
                f"模型要求特征数 {self.input_dim}，实际为 {x.shape[2]}。"
            )

        x = self.input_norm(x)
        x = self.input_projection(x)
        x = x + self.position_embedding[:, : x.shape[1], :]

        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)
        return self.head(pooled)
