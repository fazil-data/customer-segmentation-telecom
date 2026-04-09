import pandas as pd

ARPU_PREMIUM = {"HIGH", "VERY HIGH", "EXTREME"}
ARPU_MID = {"AVERAGE", "MID"}
ARPU_LOW = {"LOW", "VERY LOW"}

DATA_HEAVY = {"HIGH", "VERY HIGH", "EXTREME"}
RECENTLY_INACTIVE = {"Dormant", "Lapsed"}
RECENT_ACTIVE = {"ACTIVE", "At-Risk"}

ACTIVE_ENOUGH = {"ACTIVE", "VERY ACTIVE"}
ANY_ACTIVITY = {"ACTIVE", "VERY ACTIVE", "LESS ACTIVE"}

DECLINE_TRENDS = {"Moderate Decline", "Strong Decline"}
NEW_JOINER_FLAG = "New Joiner (Recent Only)"


def map_rating_from_percentile(percentile: float, reverse: bool = False) -> str:
    if pd.isna(percentile):
        return "UNKNOWN"

    if reverse:
        if percentile >= 95:
            return "Dormant"
        if percentile >= 75:
            return "Lapsed"
        if percentile >= 50:
            return "At-Risk"
        if percentile >= 25:
            return "ACTIVE"
        return "ACTIVE"

    if percentile >= 95:
        return "EXTREME"
    if percentile >= 75:
        return "VERY HIGH"
    if percentile >= 50:
        return "HIGH"
    if percentile >= 25:
        return "MID"
    if percentile >= 10:
        return "LOW"
    if percentile > 0:
        return "VERY LOW"
    return "DORMANT"


def add_rating_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["ARPU_RATING"] = out["mean_arpu_active__median__pct"].apply(map_rating_from_percentile)
    out["DATA_RATING"] = out["mean_data_usage_active__median__pct"].apply(map_rating_from_percentile)
    out["VOICE_RATING"] = out["mean_voice_usage_active__median__pct"].apply(map_rating_from_percentile)

    out["ACTIVE_RATING"] = out["active_ratio__median"].apply(
        lambda x: "VERY ACTIVE" if x >= 0.8 else "ACTIVE" if x >= 0.5 else "LESS ACTIVE"
    )

    out["STREAK_RATING"] = out["recent_active_streak__median"].apply(
        lambda x: "NEW STREAK" if x <= 1 else "ESTABLISHED STREAK" if x >= 4 else "MID STREAK"
    )

    out["RECENCY_RATING"] = out["months_since_last_activity__median"].apply(
        lambda x: "Dormant" if x >= 4 else "Lapsed" if x >= 2 else "ACTIVE"
    )

    return out


def map_macro_segment(row: pd.Series) -> str:
    arpu = str(row["ARPU_RATING"]).strip()
    data = str(row["DATA_RATING"]).strip()
    activity = str(row["ACTIVE_RATING"]).strip()
    recency = str(row["RECENCY_RATING"]).strip()

    if recency in RECENTLY_INACTIVE or arpu == "DORMANT":
        return "Dormant / Inactive"

    if arpu in ARPU_PREMIUM and activity not in ACTIVE_ENOUGH:
        return "Premium - At Risk"

    if arpu in ARPU_PREMIUM and activity in ACTIVE_ENOUGH:
        return "Premium"

    if data in DATA_HEAVY and arpu not in ARPU_PREMIUM and activity in ANY_ACTIVITY:
        return "Data-First"

    if arpu in ARPU_MID and activity in ANY_ACTIVITY:
        return "Mid Value"

    if arpu in ARPU_LOW:
        return "Price Sensitive / Low Value"

    return "Mid Value"


def assign_macro_segments(df: pd.DataFrame) -> pd.DataFrame:
    out = add_rating_columns(df)
    out["macro_segment"] = out.apply(map_macro_segment, axis=1)
    return out
