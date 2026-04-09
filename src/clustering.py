import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from src.config import FEATURE_COLUMNS


def run_kmeans_clustering(
    scaled_df: pd.DataFrame,
    n_clusters: int = 20,
    random_state: int = 42,
    eval_sample: int = 10000,
) -> tuple[pd.DataFrame, KMeans, dict]:
    """
    Run KMeans clustering on prepared feature data.
    """
    x = scaled_df[FEATURE_COLUMNS].values

    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=50,
        max_iter=500,
        tol=1e-5,
        algorithm="elkan",
        random_state=random_state,
    )

    labels = model.fit_predict(x)

    clustered_df = scaled_df.copy()
    clustered_df["cluster"] = labels

    sample_size = min(len(clustered_df), eval_sample)
    sample_idx = np.random.default_rng(random_state).choice(len(clustered_df), size=sample_size, replace=False)
    x_sample = x[sample_idx]
    y_sample = labels[sample_idx]

    metrics = {
        "inertia": float(model.inertia_),
        "calinski_harabasz": float(calinski_harabasz_score(x_sample, y_sample)),
        "davies_bouldin": float(davies_bouldin_score(x_sample, y_sample)),
        "silhouette": float(silhouette_score(x_sample, y_sample)),
    }

    return clustered_df, model, metrics


def profile_clusters(
    clipped_df: pd.DataFrame,
    cluster_labels,
    quantiles=(0.10, 0.25, 0.50, 0.75, 0.90),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create interpretable cluster summaries in original feature units.
    """
    df = clipped_df.copy()
    df["cluster"] = pd.Series(cluster_labels, index=df.index)

    feature_cols = [col for col in df.columns if col not in ("customer_id", "cluster")]
    grouped = df.groupby("cluster", observed=True)

    size = grouped.size().rename("size")
    size_pct = (size / size.sum() * 100).rename("size_pct")

    agg_items = [("mean", "mean"), ("median", "median"), ("std", "std")]
    for q in quantiles:
        q_name = f"q{int(q * 100):02d}"
        agg_items.append((q_name, lambda x, q=q: x.quantile(q)))
    agg_items += [("min", "min"), ("max", "max")]

    stats = grouped[feature_cols].agg(agg_items)
    stats.columns = [f"{feature}__{stat}" for feature, stat in stats.columns]

    cluster_profile_df = pd.concat([size, size_pct, stats], axis=1).sort_values("size", ascending=False)
    customer_cluster_df = df.copy()

    return cluster_profile_df, customer_cluster_df
