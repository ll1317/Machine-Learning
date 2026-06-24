from __future__ import annotations

from typing import Dict, Union

import torch
from torch import nn
import torch.nn.functional as F


class MovingAverage(nn.Module):
    """
    使用复制填充实现长度不变的一维移动平均。

    输入:
        x: [B, T]

    输出:
        trend: [B, T]
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()

        if kernel_size <= 0:
            raise ValueError("kernel_size 必须大于0。")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size 必须为奇数，以保持序列长度不变。")

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(
                f"MovingAverage 输入应为 [B,T]，实际为 {tuple(x.shape)}"
            )

        x = x.unsqueeze(1)  # [B,1,T]
        x = F.pad(
            x,
            pad=(self.padding, self.padding),
            mode="replicate",
        )
        trend = F.avg_pool1d(
            x,
            kernel_size=self.kernel_size,
            stride=1,
        )
        return trend.squeeze(1)


class SeriesDecomposition(nn.Module):
    """
    DLinear 风格的加性分解：
        original = trend + residual
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.moving_average = MovingAverage(kernel_size)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_average(x)
        residual = x - trend
        return residual, trend


class ITransformerBranch(nn.Module):
    """
    iTransformer 风格的变量标记化分支。

    输入:
        x: [B, T, N]

    处理:
        [B,T,N] -> [B,N,T]
        每个变量过去T天的历史作为一个token
        Linear(T,d_model) 后在变量维进行自注意力

    输出:
        prediction: [B,H]
        target_token: [B,d_model]
    """

    def __init__(
        self,
        input_length: int,
        num_features: int,
        horizon: int,
        target_index: int = 0,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        if input_length <= 0:
            raise ValueError("input_length 必须大于0。")
        if num_features <= 0:
            raise ValueError("num_features 必须大于0。")
        if horizon <= 0:
            raise ValueError("horizon 必须大于0。")
        if not 0 <= target_index < num_features:
            raise ValueError("target_index 超出特征范围。")
        if d_model % nhead != 0:
            raise ValueError("d_model 必须能被 nhead 整除。")

        self.input_length = input_length
        self.num_features = num_features
        self.horizon = horizon
        self.target_index = target_index
        self.d_model = d_model

        # 把每个变量的完整历史序列映射为一个变量token。
        self.history_projection = nn.Linear(input_length, d_model)

        # 变量身份嵌入，使模型区分电力、天气、日历等不同变量。
        self.variable_embedding = nn.Parameter(
            torch.zeros(1, num_features, d_model)
        )
        nn.init.normal_(self.variable_embedding, mean=0.0, std=0.02)

        self.embedding_dropout = nn.Dropout(dropout)

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

        self.forecast_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(
                f"输入应为 [B,T,N]，实际为 {tuple(x.shape)}"
            )
        if x.shape[1] != self.input_length:
            raise ValueError(
                f"要求历史长度 {self.input_length}，实际为 {x.shape[1]}。"
            )
        if x.shape[2] != self.num_features:
            raise ValueError(
                f"要求特征数 {self.num_features}，实际为 {x.shape[2]}。"
            )

        variable_history = x.transpose(1, 2)  # [B,N,T]
        tokens = self.history_projection(variable_history)  # [B,N,D]
        tokens = tokens + self.variable_embedding
        tokens = self.embedding_dropout(tokens)

        encoded = self.encoder(tokens)
        target_token = encoded[:, self.target_index, :]
        prediction = self.forecast_head(target_token)

        return prediction, target_token


class DLinearTargetBranch(nn.Module):
    """
    DLinear 风格的目标变量分解分支。

    仅使用目标变量过去 input_length 天的历史：
        target_history -> residual + trend
        residual 和 trend 分别线性预测未来 horizon 天
    """

    def __init__(
        self,
        input_length: int,
        horizon: int,
        moving_average_kernel: int = 7,
    ) -> None:
        super().__init__()

        self.input_length = input_length
        self.horizon = horizon

        self.decomposition = SeriesDecomposition(
            kernel_size=moving_average_kernel
        )

        self.residual_linear = nn.Linear(input_length, horizon)
        self.trend_linear = nn.Linear(input_length, horizon)

        # 采用接近历史均值外推的稳定初始化。
        nn.init.constant_(
            self.residual_linear.weight,
            1.0 / input_length,
        )
        nn.init.constant_(
            self.trend_linear.weight,
            1.0 / input_length,
        )
        nn.init.zeros_(self.residual_linear.bias)
        nn.init.zeros_(self.trend_linear.bias)

    def forward(
        self,
        target_history: torch.Tensor,
    ) -> torch.Tensor:
        if target_history.ndim != 2:
            raise ValueError(
                "target_history 应为 [B,T]，"
                f"实际为 {tuple(target_history.shape)}"
            )
        if target_history.shape[1] != self.input_length:
            raise ValueError(
                f"要求历史长度 {self.input_length}，"
                f"实际为 {target_history.shape[1]}。"
            )

        residual, trend = self.decomposition(target_history)

        residual_forecast = self.residual_linear(residual)
        trend_forecast = self.trend_linear(trend)

        return residual_forecast + trend_forecast


class DEITransformer(nn.Module):
    """
    分解增强倒置 Transformer（DE-iTransformer）。

    三种运行模式:
        - "deitransformer": iTransformer + DLinear + 门控融合
        - "itransformer": 仅使用倒置 Transformer 分支
        - "dlinear": 仅使用分解线性分支

    默认融合:
        y = alpha * y_itransformer + (1-alpha) * y_dlinear
        alpha = sigmoid(gate_logit)
    """

    SUPPORTED_MODES = {
        "deitransformer",
        "itransformer",
        "dlinear",
    }

    def __init__(
        self,
        input_length: int,
        num_features: int,
        horizon: int,
        target_index: int = 0,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        moving_average_kernel: int = 7,
        mode: str = "deitransformer",
        gate_init: float = 0.5,
    ) -> None:
        super().__init__()

        mode = mode.lower()
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"mode 必须属于 {sorted(self.SUPPORTED_MODES)}。"
            )
        if not 0.0 < gate_init < 1.0:
            raise ValueError("gate_init 必须位于 (0,1) 内。")

        self.input_length = input_length
        self.num_features = num_features
        self.horizon = horizon
        self.target_index = target_index
        self.mode = mode

        self.itransformer = ITransformerBranch(
            input_length=input_length,
            num_features=num_features,
            horizon=horizon,
            target_index=target_index,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.dlinear = DLinearTargetBranch(
            input_length=input_length,
            horizon=horizon,
            moving_average_kernel=moving_average_kernel,
        )

        initial_logit = torch.log(
            torch.tensor(gate_init / (1.0 - gate_init))
        )
        self.gate_logit = nn.Parameter(initial_logit)

    @property
    def fusion_weight(self) -> torch.Tensor:
        """返回 iTransformer 分支当前的融合权重 alpha。"""
        return torch.sigmoid(self.gate_logit)

    def forward(
        self,
        x: torch.Tensor,
        return_components: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if x.ndim != 3:
            raise ValueError(
                f"输入应为 [B,T,N]，实际为 {tuple(x.shape)}"
            )

        target_history = x[:, :, self.target_index]

        itransformer_prediction, _ = self.itransformer(x)
        dlinear_prediction = self.dlinear(target_history)

        if self.mode == "itransformer":
            final_prediction = itransformer_prediction
        elif self.mode == "dlinear":
            final_prediction = dlinear_prediction
        else:
            alpha = self.fusion_weight
            final_prediction = (
                alpha * itransformer_prediction
                + (1.0 - alpha) * dlinear_prediction
            )

        if not return_components:
            return final_prediction

        return {
            "prediction": final_prediction,
            "itransformer_prediction": itransformer_prediction,
            "dlinear_prediction": dlinear_prediction,
            "fusion_weight": self.fusion_weight.detach(),
        }
