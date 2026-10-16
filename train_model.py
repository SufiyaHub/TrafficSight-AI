from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DATA_PATH = DATA_DIR / "traffic_data.csv"
MODEL_PATH = MODEL_DIR / "traffic_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "weather",
    "road_type",
    "area",
    "lane_count",
    "speed_limit",
    "accident_reported",
    "event_nearby",
]

CATEGORICAL_COLUMNS = ["weather", "road_type", "area"]
NUMERIC_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "lane_count",
    "speed_limit",
    "accident_reported",
    "event_nearby",
]

TARGET_COLUMN = "vehicle_count"
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

WEATHER_OPTIONS = ["Clear", "Cloudy", "Rain", "Fog", "Storm"]
ROAD_TYPES = ["Highway", "Main Road", "City Street", "Flyover", "Service Road"]
AREAS = ["Central Business District", "University Zone", "Industrial Area", "Residential Belt", "Airport Road"]

RANGE_RULES = {
    "hour": (0, 23, "between 0 and 23"),
    "day_of_week": (0, 6, "between 0 and 6"),
    "month": (1, 12, "between 1 and 12"),
    "lane_count": (1, 12, "between 1 and 12"),
    "speed_limit": (5, 140, "between 5 and 140"),
    TARGET_COLUMN: (1, None, "greater than 0"),
}
BINARY_COLUMNS = ["is_weekend", "accident_reported", "event_nearby"]


def generate_sample_dataset(path: Path = DATA_PATH, rows_per_area: int = 24 * 180) -> pd.DataFrame:
    """Create a realistic, deterministic traffic dataset for demos and retraining."""
    rng = np.random.default_rng(42)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    timestamps = pd.date_range("2026-01-01 00:00:00", periods=rows_per_area, freq="h")
    records: list[dict[str, object]] = []

    area_base = {
        "Central Business District": 620,
        "University Zone": 360,
        "Industrial Area": 430,
        "Residential Belt": 300,
        "Airport Road": 520,
    }
    road_factor = {
        "Highway": 1.25,
        "Main Road": 1.05,
        "City Street": 0.82,
        "Flyover": 1.15,
        "Service Road": 0.62,
    }
    weather_factor = {
        "Clear": 1.0,
        "Cloudy": 0.96,
        "Rain": 0.82,
        "Fog": 0.72,
        "Storm": 0.58,
    }

    for area_index, area in enumerate(AREAS):
        for ts in timestamps:
            hour = int(ts.hour)
            day_of_week = int(ts.dayofweek)
            month = int(ts.month)
            is_weekend = int(day_of_week >= 5)

            road_type = ROAD_TYPES[(hour + day_of_week + area_index) % len(ROAD_TYPES)]
            weather = rng.choice(WEATHER_OPTIONS, p=[0.58, 0.22, 0.12, 0.06, 0.02])
            lane_count = int(rng.choice([2, 3, 4, 5, 6], p=[0.18, 0.34, 0.26, 0.14, 0.08]))
            speed_limit = int(rng.choice([30, 40, 50, 60, 70, 80], p=[0.08, 0.18, 0.24, 0.22, 0.18, 0.10]))

            morning_peak = np.exp(-((hour - 8) / 2.5) ** 2) * 260
            evening_peak = np.exp(-((hour - 18) / 3.0) ** 2) * 310
            lunch_bump = np.exp(-((hour - 13) / 2.8) ** 2) * 70
            weekend_drop = -130 if is_weekend else 0
            month_wave = np.sin((month / 12) * 2 * np.pi) * 45

            event_nearby = int(rng.random() < (0.05 + (0.08 if area in ["University Zone", "Airport Road"] else 0.0)))
            accident_reported = int(rng.random() < (0.025 + (0.025 if weather in ["Rain", "Fog", "Storm"] else 0.0)))

            count = (
                area_base[area]
                * road_factor[road_type]
                * weather_factor[str(weather)]
                + morning_peak
                + evening_peak
                + lunch_bump
                + weekend_drop
                + lane_count * 42
                - speed_limit * 1.7
                + event_nearby * 165
                - accident_reported * 95
                + month_wave
                + rng.normal(0, 55)
            )

            records.append(
                {
                    "timestamp": ts,
                    "hour": hour,
                    "day_of_week": day_of_week,
                    "month": month,
                    "is_weekend": is_weekend,
                    "weather": str(weather),
                    "road_type": road_type,
                    "area": area,
                    "lane_count": lane_count,
                    "speed_limit": speed_limit,
                    "accident_reported": accident_reported,
                    "event_nearby": event_nearby,
                    "vehicle_count": max(45, int(round(count))),
                }
            )

    df = pd.DataFrame(records)
    df.to_csv(path, index=False)
    return df


def _coerce_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for column in NUMERIC_COLUMNS + [TARGET_COLUMN]:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    for column in CATEGORICAL_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].astype("string").str.strip()
    return cleaned


def validate_training_data(df: pd.DataFrame) -> list[str]:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    errors = [f"Missing required column: {column}" for column in missing]

    if not errors:
        cleaned = _coerce_training_frame(df)

        for column in NUMERIC_COLUMNS + [TARGET_COLUMN]:
            blank_count = int(cleaned[column].isna().sum())
            if blank_count:
                errors.append(f"Column has {blank_count} missing or non-numeric value(s): {column}")

        for column, (minimum, maximum, description) in RANGE_RULES.items():
            if cleaned[column].isna().any():
                continue
            invalid = cleaned[column] < minimum
            if maximum is not None:
                invalid = invalid | (cleaned[column] > maximum)
            if bool(invalid.any()):
                errors.append(f"Column values must be {description}: {column}")

        for column in ["hour", "day_of_week", "month", "lane_count"] + BINARY_COLUMNS:
            if cleaned[column].isna().any():
                continue
            non_integer = cleaned[column] % 1 != 0
            if bool(non_integer.any()):
                errors.append(f"Column must contain whole numbers: {column}")

        for column in BINARY_COLUMNS:
            if cleaned[column].isna().any():
                continue
            invalid_binary = ~cleaned[column].isin([0, 1])
            if bool(invalid_binary.any()):
                errors.append(f"Column must contain only 0 or 1 values: {column}")

        for column in CATEGORICAL_COLUMNS:
            if cleaned[column].isna().any() or bool((cleaned[column] == "").any()):
                errors.append(f"Column contains blank values: {column}")

        if len(df) < 100:
            errors.append("Upload at least 100 rows so the model can be evaluated reliably.")

    return errors


def training_data_warnings(df: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        warnings.append(f"{duplicate_count:,} duplicate row(s) were detected and will be kept for training.")

    extra_columns = [column for column in df.columns if column not in REQUIRED_COLUMNS + ["timestamp", "traffic_level", "day_name"]]
    if extra_columns:
        warnings.append(f"Extra column(s) will be ignored by the model: {', '.join(extra_columns)}")

    return warnings


def _first_date_for_month_weekday(year: int, month: int, day_of_week: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    offset = (day_of_week - int(first.dayofweek)) % 7
    return first + pd.Timedelta(days=offset)


def _timestamps_from_features(df: pd.DataFrame) -> pd.Series:
    occurrences: dict[tuple[int, int, int], int] = {}
    generated: list[pd.Timestamp] = []

    for row in df.itertuples(index=False):
        month = int(getattr(row, "month"))
        day_of_week = int(getattr(row, "day_of_week"))
        hour = int(getattr(row, "hour"))
        key = (month, day_of_week, hour)
        occurrence = occurrences.get(key, 0)
        occurrences[key] = occurrence + 1

        base_date = _first_date_for_month_weekday(2026, month, day_of_week)
        generated.append(base_date + pd.Timedelta(days=7 * (occurrence % 4), hours=hour))

    return pd.Series(generated, index=df.index)


def prepare_training_data(df: pd.DataFrame) -> pd.DataFrame:
    errors = validate_training_data(df)
    if errors:
        raise ValueError("; ".join(errors))

    cleaned = _coerce_training_frame(df)
    for column in ["hour", "day_of_week", "month", "lane_count"] + BINARY_COLUMNS:
        cleaned[column] = cleaned[column].astype(int)
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(float)
    return cleaned[REQUIRED_COLUMNS].copy()


def prepare_app_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = prepare_training_data(df)
    if "timestamp" in df.columns:
        timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
        if timestamps.isna().any():
            timestamps = _timestamps_from_features(cleaned)
    else:
        timestamps = _timestamps_from_features(cleaned)

    cleaned.insert(0, "timestamp", timestamps)
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].round().astype(int)
    cleaned["traffic_level"] = cleaned[TARGET_COLUMN].apply(_traffic_level_label)
    cleaned["day_name"] = cleaned["day_of_week"].map(
        {
            0: "Monday",
            1: "Tuesday",
            2: "Wednesday",
            3: "Thursday",
            4: "Friday",
            5: "Saturday",
            6: "Sunday",
        }
    )
    return cleaned


def _traffic_level_label(vehicle_count: Any) -> str:
    value = float(vehicle_count)
    if value < 450:
        return "Low"
    if value < 750:
        return "Moderate"
    if value < 1050:
        return "High"
    return "Congested"


def _build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_COLUMNS),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLUMNS),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=220,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def _feature_importance(pipeline: Pipeline) -> list[dict[str, float | str]]:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    clean_names = [name.replace("numeric__", "").replace("categorical__", "") for name in names]
    importances = model.feature_importances_
    pairs = sorted(zip(clean_names, importances), key=lambda item: item[1], reverse=True)
    return [{"feature": feature, "importance": round(float(value), 5)} for feature, value in pairs[:18]]


def train_and_save(df: pd.DataFrame | None = None) -> dict[str, object]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if df is None:
        df = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else generate_sample_dataset(DATA_PATH)
    else:
        df = df.copy()

    training_df = prepare_training_data(df)

    X = training_df[FEATURE_COLUMNS]
    y = training_df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=42)

    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = r2_score(y_test, predictions)
    mape = float(np.mean(np.abs((y_test - predictions) / np.maximum(y_test, 1))) * 100)

    eval_df = pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "predicted": np.round(predictions, 1),
            "error": np.round(y_test.to_numpy() - predictions, 1),
        }
    ).sample(min(350, len(y_test)), random_state=42)

    metrics: dict[str, object] = {
        "rows": int(len(training_df)),
        "trained_on": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "mae": round(float(mae), 2),
        "rmse": round(rmse, 2),
        "r2": round(float(r2), 4),
        "mape": round(mape, 2),
        "data_warnings": training_data_warnings(df),
        "feature_importance": _feature_importance(pipeline),
        "evaluation_sample": eval_df.to_dict(orient="records"),
    }

    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    prepare_app_dataset(df).to_csv(DATA_PATH, index=False)
    return metrics


if __name__ == "__main__":
    if not DATA_PATH.exists():
        generate_sample_dataset(DATA_PATH)
    result = train_and_save()
    print(f"Trained TrafficSight AI model on {result['rows']} rows")
    print(f"MAE: {result['mae']} | RMSE: {result['rmse']} | R2: {result['r2']}")
