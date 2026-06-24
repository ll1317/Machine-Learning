import pandas as pd
from pathlib import Path

DATA_DIR = Path(r"D:\\Users\\ll\\桌面\\机器学习\\data")
DAILY_PATH = DATA_DIR / "daily_power.csv"
WEATHER_PATH = DATA_DIR / "weather_data.csv"

MERGED_PATH = DATA_DIR / "daily_power_weather.csv"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

STATION_ID = "75114001"
WEATHER_COLUMNS = [
    "RR",
    "NBJRR1",
    "NBJRR5",
    "NBJRR10",
    "NBJBROU",
]

daily = pd.read_csv(DAILY_PATH, parse_dates=["date"])

weather = pd.read_csv(
    WEATHER_PATH,
    sep=";",
    encoding="utf-8",
    dtype={"NUM_POSTE": str},
    low_memory=False,
)

weather["AAAAMM"] = pd.to_numeric(
    weather["AAAAMM"],
    errors="coerce",
)

weather = weather[
    (weather["NUM_POSTE"] == STATION_ID)
    & weather["AAAAMM"].between(200612, 201011)
].copy()

weather["year_month"] = pd.to_datetime(
    weather["AAAAMM"].astype("Int64").astype(str),
    format="%Y%m",
).dt.strftime("%Y-%m")

weather = weather[
    [
        "year_month",
        "NUM_POSTE",
        "NOM_USUEL",
        *WEATHER_COLUMNS,
    ]
].rename(
    columns={
        "NUM_POSTE": "weather_station_id",
        "NOM_USUEL": "weather_station_name",
    }
)

if weather["year_month"].nunique() != 48:
    raise ValueError(
        "PARIS-MONTSOURIS 在目标时间范围内没有完整的48个月数据。"
    )

merged = daily.merge(
    weather,
    on="year_month",
    how="left",
    validate="many_to_one",
)

if merged[WEATHER_COLUMNS].isna().any().any():
    raise ValueError("合并后仍有天气字段缺失，请检查月份或站点。")

merged.to_csv(
    MERGED_PATH,
    index=False,
    encoding="utf-8-sig",
)

split_date = pd.Timestamp("2009-11-27")
train = merged[merged["date"] < split_date].copy()
test = merged[merged["date"] >= split_date].copy()

train.to_csv(TRAIN_PATH, index=False, encoding="utf-8-sig")
test.to_csv(TEST_PATH, index=False, encoding="utf-8-sig")

print("合并文件：", MERGED_PATH)
print("合并数据：", len(merged), "行")
print("训练集：", len(train), "行，",
      train["date"].min().date(), "至", train["date"].max().date())
print("测试集：", len(test), "行，",
      test["date"].min().date(), "至", test["date"].max().date())
print("测试输入90天：2009-08-29 至 2009-11-26")
print("短期目标90天：2009-11-27 至 2010-02-24")
print("长期目标365天：2009-11-27 至 2010-11-26")
