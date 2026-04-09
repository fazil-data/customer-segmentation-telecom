import numpy as np
import pandas as pd


FEATURES_MAP = {
    "mean_arpu_active__median": "mean_arpu_active",
    "mean_data_usage_active__median": "mean_data_usage_active",
    "mean_voice_usage_active__median": "mean_voice_usage_active",
    "active_ratio__median": "active_ratio",
    "recent_active_streak__median": "recent_active_streak",
    "months_since_last_activity__median": "months_since_last_activity",
}


def value_to_percentile(value, base_series: pd.Series) -> float:
    series = pd.to_numeric(base_series, errors="coerce").dropna().values
    if series.size == 0 or pd.isna(value):
        return np.nan
    series.sort(kind="mergesort")
    position = np.searchsorted(series, float(value), side="right")
    return 100.0 * position / series.size


def annotate_cluster_with_percentiles(
    cluster_profile_df: pd.DataFrame,
    clipped_df: pd.DataFrame,
    features_map: dict = FEATURES_MAP,
) -> pd.DataFrame:
    out = cluster_profile_df.copy()

    for cluster_col, base_col in features_map.items():
        if cluster_col not in out.columns or base_col not in clipped_df.columns:
            continue

        percentiles = out[cluster_col].apply(lambda value: value_to_percentile(value, clipped_df[base_col]))
        out[f"{cluster_col}__pct"] = percentiles
        out[f"{cluster_col}__P"] = percentiles.apply(lambda x: f"P{int(round(x))}" if pd.notna(x) else "PNA")

    return out


def format_data_usage_mb(value: float) -> str:
    if pd.isna(value):
        return "0 MB"
    if value >= 1024:
        return f"{value / 1024:.1f} GB"
    if value >= 1:
        return f"{value:.0f} MB"
    return f"{value:.3f} MB"


def build_cluster_description(row: pd.Series) -> str:
    arpu = row.get("mean_arpu_active__median", np.nan)
    data = row.get("mean_data_usage_active__median", np.nan)
    voice = row.get("mean_voice_usage_active__median", np.nan)
    active_ratio = row.get("active_ratio__median", np.nan)
    streak = row.get("recent_active_streak__median", np.nan)
    recency = row.get("months_since_last_activity__median", np.nan)

    arpu_p = row.get("mean_arpu_active__median__P", "PNA")
    data_p = row.get("mean_data_usage_active__median__P", "PNA")
    voice_p = row.get("mean_voice_usage_active__median__P", "PNA")
    active_p = row.get("active_ratio__median__P", "PNA")

    return (
        f"ARPU ~ {round(arpu, 2) if pd.notna(arpu) else 0} ({arpu_p}) | "
        f"Data ~ {format_data_usage_mb(data)} ({data_p}) | "
        f"Voice ~ {round(voice, 1) if pd.notna(voice) else 0} mins ({voice_p}) | "
        f"Active ratio ~ {round(active_ratio, 2) if pd.notna(active_ratio) else 0} ({active_p}) | "
        f"Recent streak ~ {int(streak) if pd.notna(streak) else 0} | "
        f"Months since last activity ~ {int(recency) if pd.notna(recency) else 0}"
    )


def add_cluster_descriptions(annotated_df: pd.DataFrame) -> pd.DataFrame:
    out = annotated_df.copy()
    out["cluster_description"] = out.apply(build_cluster_description, axis=1)
    return out
