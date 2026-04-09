from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

FEATURE_COLUMNS = [
    "mean_arpu_active",
    "mean_data_usage_active",
    "mean_voice_usage_active",
    "active_ratio",
    "std_data_usage_active",
    "std_arpu_active",
    "data_trend",
    "arpu_trend",
    "recent_active_streak",
    "months_since_last_activity",
]

CLIP_PERCENTILES = {
    "mean_arpu_active": (0.01, 0.99),
    "mean_data_usage_active": (0.01, 0.99),
    "mean_voice_usage_active": (0.01, 0.99),
    "active_ratio": None,
    "std_data_usage_active": (0.01, 0.99),
    "std_arpu_active": (0.01, 0.99),
    "data_trend": (0.02, 0.98),
    "arpu_trend": (0.02, 0.98),
    "recent_active_streak": None,
    "months_since_last_activity": None,
}
