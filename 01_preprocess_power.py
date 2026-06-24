from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(r"D:\\Users\\ll\\桌面\\机器学习\\data")
POWER_PATH = DATA_DIR / "individual+household+electric+power+consumption\\household_power_consumption.txt"
OUTPUT_PATH = DATA_DIR / "daily_power.csv"

NUMERIC_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


def main() -> None:
    if not POWER_PATH.exists():
        raise FileNotFoundError(f"找不到电力数据：{POWER_PATH}")

    print("正在读取分钟级电力数据……")

    df = pd.read_csv(
        POWER_PATH,
        sep=";",
        na_values=["?", ""],
        low_memory=False,
    )

    print("原始数据形状：", df.shape)
    print("原始列名：", df.columns.tolist())

    # 合并日期和时间
    df["datetime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )

    df = df.dropna(subset=["datetime"])

    # 将电力字段转为数值
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = (
        df.drop(columns=["Date", "Time"])
        .set_index("datetime")
        .sort_index()
    )

    print("\n分钟级缺失值数量：")
    print(df[NUMERIC_COLUMNS].isna().sum())

    # 使用时间插值处理少量缺失值
    df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].interpolate(
        method="time",
        limit_direction="both",
    )

    # 按老师要求汇总为每日数据
    daily = df.resample("D").agg(
        {
            "Global_active_power": "sum",
            "Global_reactive_power": "sum",
            "Voltage": "mean",
            "Global_intensity": "mean",
            "Sub_metering_1": "sum",
            "Sub_metering_2": "sum",
            "Sub_metering_3": "sum",
        }
    )

    daily = daily.rename(
        columns={
            "Global_active_power": "global_active_power",
            "Global_reactive_power": "global_reactive_power",
            "Voltage": "voltage",
            "Global_intensity": "global_intensity",
            "Sub_metering_1": "sub_metering_1",
            "Sub_metering_2": "sub_metering_2",
            "Sub_metering_3": "sub_metering_3",
        }
    )

    # 计算未被三个分表覆盖的剩余能耗
    daily["sub_metering_remainder"] = (
        daily["global_active_power"] * 1000 / 60
        - daily["sub_metering_1"]
        - daily["sub_metering_2"]
        - daily["sub_metering_3"]
    )

    daily = daily.reset_index().rename(columns={"datetime": "date"})

    # 增加年月，后面用于合并月度天气
    daily["year_month"] = daily["date"].dt.strftime("%Y-%m")

    # 增加时间特征
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(int)
    daily["month"] = daily["date"].dt.month
    daily["day_of_year"] = daily["date"].dt.dayofyear

    daily["dow_sin"] = np.sin(
        2 * np.pi * daily["day_of_week"] / 7
    )
    daily["dow_cos"] = np.cos(
        2 * np.pi * daily["day_of_week"] / 7
    )
    daily["month_sin"] = np.sin(
        2 * np.pi * daily["month"] / 12
    )
    daily["month_cos"] = np.cos(
        2 * np.pi * daily["month"] / 12
    )

    # 删除仍然存在严重缺失的日记录
    daily = daily.dropna().reset_index(drop=True)

    daily.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n处理完成。")
    print("输出文件：", OUTPUT_PATH)
    print("日级数据形状：", daily.shape)
    print(
        "日期范围：",
        daily["date"].min(),
        "至",
        daily["date"].max(),
    )
    print("\n前5行：")
    print(daily.head().to_string())


if __name__ == "__main__":
    main()