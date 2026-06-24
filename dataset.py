from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


# 三种模型应使用完全相同的输入特征，保证比较公平。
FEATURE_COLUMNS: List[str] = [
    "global_active_power",
    "global_reactive_power",
    "voltage",
    "global_intensity",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
    "sub_metering_remainder",
    "is_weekend",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "RR",
    "NBJRR1",
    "NBJRR5",
    "NBJRR10",
    "NBJBROU",
]

TARGET_COLUMN = "global_active_power"
INPUT_LENGTH = 90
SUPPORTED_HORIZONS = (90, 365)


class PowerForecastDataset(Dataset):
    """用于多变量时间序列直接多步预测的数据集。"""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        if x.ndim != 3:
            raise ValueError(f"x 应为 [N, input_len, features]，实际为 {x.shape}")
        if y.ndim != 2:
            raise ValueError(f"y 应为 [N, horizon]，实际为 {y.shape}")
        if len(x) != len(y):
            raise ValueError("x 与 y 的样本数量不一致。")

        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


@dataclass
class TaskDataBundle:
    """一个预测长度对应的全部数据、标准化器和日期信息。"""

    horizon: int
    input_length: int
    feature_columns: List[str]
    target_column: str

    train_dataset: PowerForecastDataset
    val_dataset: PowerForecastDataset
    test_dataset: PowerForecastDataset

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader

    feature_scaler: StandardScaler
    target_scaler: StandardScaler

    date_ranges: Dict[str, Tuple[str, str]]

    def inverse_transform_target(self, values: np.ndarray) -> np.ndarray:
        """将模型输出从标准化尺度恢复为原始 global_active_power 尺度。"""
        array = np.asarray(values, dtype=np.float64)
        original_shape = array.shape
        restored = self.target_scaler.inverse_transform(array.reshape(-1, 1))
        return restored.reshape(original_shape)


def set_seed(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件：{path}")

    frame = pd.read_csv(path, parse_dates=["date"])
    frame = frame.sort_values("date").reset_index(drop=True)

    required_columns = {"date", TARGET_COLUMN, *FEATURE_COLUMNS}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} 缺少字段：{sorted(missing)}")

    if frame["date"].duplicated().any():
        duplicate_dates = frame.loc[frame["date"].duplicated(), "date"].tolist()
        raise ValueError(f"{path.name} 存在重复日期：{duplicate_dates[:5]}")

    expected_dates = pd.date_range(
        frame["date"].min(),
        frame["date"].max(),
        freq="D",
    )
    if len(expected_dates) != len(frame) or not np.array_equal(
        expected_dates.to_numpy(),
        frame["date"].to_numpy(),
    ):
        raise ValueError(f"{path.name} 的日期不是连续的每日序列。")

    numeric_columns = list(dict.fromkeys(FEATURE_COLUMNS + [TARGET_COLUMN]))
    frame[numeric_columns] = frame[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if frame[numeric_columns].isna().any().any():
        bad_columns = frame[numeric_columns].columns[
            frame[numeric_columns].isna().any()
        ].tolist()
        raise ValueError(f"{path.name} 仍有缺失或非数值字段：{bad_columns}")

    return frame


def _scale_target(
    values: np.ndarray,
    scaler: StandardScaler,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    original_shape = values.shape
    scaled = scaler.transform(values.reshape(-1, 1))
    return scaled.reshape(original_shape)


def _create_windows(
    features: np.ndarray,
    targets: np.ndarray,
    input_length: int,
    horizon: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """从连续训练区间生成 [过去90天 -> 未来H天] 的滑动窗口。"""
    if stride < 1:
        raise ValueError("stride 必须大于等于1。")

    total_length = input_length + horizon
    if len(features) < total_length:
        raise ValueError(
            f"训练区间只有 {len(features)} 天，"
            f"不足以生成 input={input_length}, horizon={horizon} 的样本。"
        )

    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    last_start = len(features) - total_length
    for start in range(0, last_start + 1, stride):
        input_end = start + input_length
        output_end = input_end + horizon
        x_list.append(features[start:input_end])
        y_list.append(targets[input_end:output_end])

    return (
        np.asarray(x_list, dtype=np.float32),
        np.asarray(y_list, dtype=np.float32),
    )


def _single_forecast_sample(
    input_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    feature_scaler: StandardScaler,
    target_scaler: StandardScaler,
) -> Tuple[np.ndarray, np.ndarray]:
    x = feature_scaler.transform(
        input_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    )
    y = _scale_target(
        target_frame[TARGET_COLUMN].to_numpy(dtype=np.float64),
        target_scaler,
    )
    return x[np.newaxis, ...].astype(np.float32), y[np.newaxis, ...].astype(
        np.float32
    )


def prepare_task_data(
    train_csv: str | Path,
    test_csv: str | Path,
    horizon: int,
    batch_size: int = 32,
    stride: int = 1,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool | None = None,
) -> TaskDataBundle:
    """
    构造一个预测任务的数据。

    划分原则：
    1. test.csv 只用于最终测试目标；
    2. train.csv 最后 horizon 天作为验证目标；
    3. 验证目标之前的90天作为验证输入；
    4. train.csv 最后90天作为最终测试输入；
    5. 标准化器仅在验证目标开始之前的训练区间上拟合。
    """
    if horizon not in SUPPORTED_HORIZONS:
        raise ValueError(
            f"horizon 只支持 {SUPPORTED_HORIZONS}，实际收到 {horizon}。"
        )

    set_seed(seed)

    train_frame = _load_csv(Path(train_csv))
    test_frame = _load_csv(Path(test_csv))

    if len(test_frame) < horizon:
        raise ValueError(
            f"test.csv 只有 {len(test_frame)} 天，不能评估未来 {horizon} 天。"
        )

    validation_target_start = len(train_frame) - horizon
    validation_input_start = validation_target_start - INPUT_LENGTH

    if validation_input_start < 0:
        raise ValueError(
            "train.csv 太短，无法同时构造验证输入和验证目标。"
        )

    # 验证目标之前的全部历史均属于训练可用历史。
    training_history = train_frame.iloc[:validation_target_start].copy()

    validation_input = train_frame.iloc[
        validation_input_start:validation_target_start
    ].copy()
    validation_target = train_frame.iloc[validation_target_start:].copy()

    test_input = train_frame.iloc[-INPUT_LENGTH:].copy()
    test_target = test_frame.iloc[:horizon].copy()

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    feature_scaler.fit(
        training_history[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    )
    target_scaler.fit(
        training_history[[TARGET_COLUMN]].to_numpy(dtype=np.float64)
    )

    scaled_train_features = feature_scaler.transform(
        training_history[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    )
    scaled_train_targets = _scale_target(
        training_history[TARGET_COLUMN].to_numpy(dtype=np.float64),
        target_scaler,
    )

    x_train, y_train = _create_windows(
        features=scaled_train_features,
        targets=scaled_train_targets,
        input_length=INPUT_LENGTH,
        horizon=horizon,
        stride=stride,
    )

    x_val, y_val = _single_forecast_sample(
        validation_input,
        validation_target,
        feature_scaler,
        target_scaler,
    )
    x_test, y_test = _single_forecast_sample(
        test_input,
        test_target,
        feature_scaler,
        target_scaler,
    )

    train_dataset = PowerForecastDataset(x_train, y_train)
    val_dataset = PowerForecastDataset(x_val, y_val)
    test_dataset = PowerForecastDataset(x_test, y_test)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    date_ranges = {
        "training_history": (
            training_history["date"].iloc[0].strftime("%Y-%m-%d"),
            training_history["date"].iloc[-1].strftime("%Y-%m-%d"),
        ),
        "validation_input": (
            validation_input["date"].iloc[0].strftime("%Y-%m-%d"),
            validation_input["date"].iloc[-1].strftime("%Y-%m-%d"),
        ),
        "validation_target": (
            validation_target["date"].iloc[0].strftime("%Y-%m-%d"),
            validation_target["date"].iloc[-1].strftime("%Y-%m-%d"),
        ),
        "test_input": (
            test_input["date"].iloc[0].strftime("%Y-%m-%d"),
            test_input["date"].iloc[-1].strftime("%Y-%m-%d"),
        ),
        "test_target": (
            test_target["date"].iloc[0].strftime("%Y-%m-%d"),
            test_target["date"].iloc[-1].strftime("%Y-%m-%d"),
        ),
    }

    return TaskDataBundle(
        horizon=horizon,
        input_length=INPUT_LENGTH,
        feature_columns=FEATURE_COLUMNS.copy(),
        target_column=TARGET_COLUMN,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        date_ranges=date_ranges,
    )


def _print_bundle_summary(bundle: TaskDataBundle) -> None:
    print("=" * 72)
    print(f"预测任务：过去 {bundle.input_length} 天预测未来 {bundle.horizon} 天")
    print(f"输入特征数：{len(bundle.feature_columns)}")
    print(f"训练样本数：{len(bundle.train_dataset)}")
    print(f"验证样本数：{len(bundle.val_dataset)}")
    print(f"测试样本数：{len(bundle.test_dataset)}")

    x_train, y_train = bundle.train_dataset[0]
    x_val, y_val = bundle.val_dataset[0]
    x_test, y_test = bundle.test_dataset[0]

    print(f"训练单样本形状：X={tuple(x_train.shape)}, y={tuple(y_train.shape)}")
    print(f"验证单样本形状：X={tuple(x_val.shape)}, y={tuple(y_val.shape)}")
    print(f"测试单样本形状：X={tuple(x_test.shape)}, y={tuple(y_test.shape)}")

    print("\n日期范围：")
    for name, (start, end) in bundle.date_ranges.items():
        print(f"  {name:18s}: {start} -> {end}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="构造家庭电力90天/365天预测数据集。"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(r"D:\\Users\\ll\\桌面\\机器学习\\data"),
        help="train.csv 和 test.csv 所在目录。"
    )
    parser.add_argument(
        "--horizon",
        type=str,
        default="both",
        choices=["90", "365", "both"],
        help="预测长度。",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    train_csv = args.data_dir / "train.csv"
    test_csv = args.data_dir / "test.csv"

    horizons = SUPPORTED_HORIZONS if args.horizon == "both" else (
        int(args.horizon),
    )

    for horizon in horizons:
        bundle = prepare_task_data(
            train_csv=train_csv,
            test_csv=test_csv,
            horizon=horizon,
            batch_size=args.batch_size,
            stride=args.stride,
            seed=args.seed,
            num_workers=args.num_workers,
        )
        _print_bundle_summary(bundle)


if __name__ == "__main__":
    main()
