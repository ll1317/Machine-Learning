from __future__ import annotations

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    """
    多变量 LSTM 直接多步预测模型。

    输入:
        x: [batch_size, input_length, input_dim]

    输出:
        y_hat: [batch_size, horizon]
    """

    def __init__(
        self,
        input_dim: int,
        horizon: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim 必须大于 0。")
        if horizon <= 0:
            raise ValueError("horizon 必须大于 0。")
        if hidden_size <= 0:
            raise ValueError("hidden_size 必须大于 0。")
        if num_layers <= 0:
            raise ValueError("num_layers 必须大于 0。")

        # 当 LSTM 只有一层时，PyTorch 的循环层 dropout 不生效。
        recurrent_dropout = dropout if num_layers > 1 else 0.0

        self.input_dim = input_dim
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_norm = nn.LayerNorm(input_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )

        # 使用最后一个时间步的隐状态，一次性输出未来 horizon 天。
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"输入应为三维张量 [B, T, F]，实际形状为 {tuple(x.shape)}"
            )

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"模型要求 {self.input_dim} 个特征，实际输入 {x.shape[-1]} 个。"
            )

        x = self.input_norm(x)
        sequence_output, _ = self.lstm(x)
        last_hidden = sequence_output[:, -1, :]
        return self.head(last_hidden)
