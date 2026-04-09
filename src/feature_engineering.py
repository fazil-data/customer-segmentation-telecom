import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import FEATURE_COLUMNS, CLIP_PERCENTILES


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build behavioural features from 6 months of customer activity.
    Expected columns:
      customer_id
      m1_arpu ... m6_arpu
      m1_data_usage ... m6_data_usage
      m1_voice_usage ... m6_voice_usage
      m1_active ... m6_active
    """
    required = ["customer_id"] + [
        f"m{k}_{col}"
        for k in range(1, 7)
        for col in ("arpu", "data_usage", "voice_usage", "active")
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    work_df = df.copy()

    month_cols = [col for col in work_df.columns if col.startswith(("m1_", "m2_", "m3_", "m4_", "m5_", "m6_"))]
    work_df[month_cols] = work_df[month_cols].fillna(0)

    months = np.arange(1, 7, dtype=float)
    arpu = work_df[[f"m{k}_arpu" for k in range(1, 7)]].to_numpy(dtype=float)
    data = work_df[[f"m{k}_data_usage" for k in range(1, 7)]].to_numpy(dtype=float)
    voice = work_df[[f"m{k}_voice_usage" for k in range(1, 7)]].to_numpy(dtype=float)
    active = work_df[[f"m{k}_active" for k in range(1, 7)]].to_numpy(dtype=float)

    active_bool = active.astype(bool)
    n_active = active.sum(axis=1)
    active_ratio = n_active / 6.0

    def mean_active(values: np.ndarray) -> np.ndarray:
        sums = (values * active_bool).sum(axis=1)
        denom = np.maximum(n_active, 1)
        output = sums / denom
        output[n_active == 0] = 0.0
        return output

    def std_active(values: np.ndarray) -> np.ndarray:
        masked = np.where(active_bool, values, np.nan)
        output = np.nanstd(masked, axis=1, ddof=0)
        return np.where(np.isnan(output), 0.0, output)

    def slope_ols(values: np.ndarray) -> np.ndarray:
        x = months.reshape(1, -1)
        x2 = (months ** 2).reshape(1, -1)

        sx = (x * active_bool).sum(axis=1)
        sy = (values * active_bool).sum(axis=1)
        sxx = (x2 * active_bool).sum(axis=1)
        sxy = (x * values * active_bool).sum(axis=1)

        numerator = n_active * sxy - sx * sy
        denominator = n_active * sxx - sx * sx
        denominator = np.where(denominator == 0, np.nan, denominator)
        return numerator / denominator

    def ratio_fallback_row(values: np.ndarray, flags: np.ndarray) -> float:
        active_idx = [i for i, is_active in enumerate(flags, start=1) if is_active]
        if not active_idx:
            return 1.0

        first_idx = active_idx[:2]
        last_idx = active_idx[-2:]

        first_mean = np.mean([values[i - 1] for i in first_idx]) if first_idx else np.nan
        last_mean = np.mean([values[i - 1] for i in last_idx]) if last_idx else np.nan

        if pd.isna(first_mean) or first_mean == 0:
            return 1.0

        return float(last_mean / first_mean)

    arpu_trend = slope_ols(arpu)
    data_trend = slope_ols(data)

    need_ratio_fallback = n_active < 3
    if need_ratio_fallback.any():
        rows = np.flatnonzero(need_ratio_fallback)
        arpu_trend[need_ratio_fallback] = [ratio_fallback_row(arpu[i, :], active_bool[i, :]) for i in rows]
        data_trend[need_ratio_fallback] = [ratio_fallback_row(data[i, :], active_bool[i, :]) for i in rows]

    arpu_trend = np.where(n_active == 0, 0.0, arpu_trend)
    data_trend = np.where(n_active == 0, 0.0, data_trend)
    arpu_trend = np.nan_to_num(arpu_trend, nan=0.0)
    data_trend = np.nan_to_num(data_trend, nan=0.0)

    def recent_streak_row(flags: np.ndarray) -> int:
        streak = 0
        for value in flags[::-1]:
            if value:
                streak += 1
            else:
                break
        return streak

    def recency_row(flags: np.ndarray) -> int:
        active_idx = [i for i, is_active in enumerate(flags, start=1) if is_active]
        if not active_idx:
            return 6
        last_active = active_idx[-1]
        return 6 - last_active

    recent_active_streak = np.fromiter(
        (recent_streak_row(row) for row in active_bool),
        dtype=int,
        count=active_bool.shape[0],
    )

    months_since_last_activity = np.fromiter(
        (recency_row(row) for row in active_bool),
        dtype=int,
        count=active_bool.shape[0],
    )

    return pd.DataFrame(
        {
            "customer_id": work_df["customer_id"].values,
            "mean_arpu_active": mean_active(arpu),
            "mean_data_usage_active": mean_active(data),
            "mean_voice_usage_active": mean_active(voice),
            "active_ratio": active_ratio,
            "std_data_usage_active": std_active(data),
            "std_arpu_active": std_active(arpu),
            "data_trend": data_trend,
            "arpu_trend": arpu_trend,
            "recent_active_streak": recent_active_streak,
            "months_since_last_activity": months_since_last_activity,
        }
    )


def prepare_features_for_clustering(features_df: pd.DataFrame):
    """
    Clip selected features and standardize them for clustering.
    """
    x = features_df[FEATURE_COLUMNS].copy()
    clip_bounds = {}

    for col in FEATURE_COLUMNS:
        bounds = CLIP_PERCENTILES.get(col)
        if bounds is None:
            clip_bounds[col] = None
            continue

        low_pct, high_pct = bounds
        low_value = x[col].quantile(low_pct)
        high_value = x[col].quantile(high_pct)
        x[col] = x[col].clip(low_value, high_value)
        clip_bounds[col] = (float(low_value), float(high_value))

    clipped_df = pd.concat([features_df[["customer_id"]], x], axis=1)

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(x.values)

    scaled_df = pd.DataFrame(scaled_values, columns=FEATURE_COLUMNS, index=features_df.index)
    scaled_df.insert(0, "customer_id", features_df["customer_id"].values)

    return clipped_df, scaled_df, scaler, clip_bounds
