import pandas as pd

from src.feature_engineering import build_customer_features, prepare_features_for_clustering
from src.clustering import run_kmeans_clustering, profile_clusters
from src.cluster_annotation import annotate_cluster_with_percentiles, add_cluster_descriptions
from src.macro_mapping import assign_macro_segments
from src.config import OUTPUT_DIR

INPUT_FILE = "data/sample_customer_usage_6m.csv"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(INPUT_FILE)

    features_df = build_customer_features(raw_df)
    clipped_df, scaled_df, _, _ = prepare_features_for_clustering(features_df)

    clustered_df, model, metrics = run_kmeans_clustering(
        scaled_df=scaled_df,
        n_clusters=20,
        random_state=42,
    )

    cluster_profile_df, customer_cluster_df = profile_clusters(
        clipped_df=clipped_df,
        cluster_labels=clustered_df["cluster"],
    )

    annotated_df = annotate_cluster_with_percentiles(cluster_profile_df, clipped_df)
    described_df = add_cluster_descriptions(annotated_df)
    final_cluster_book = assign_macro_segments(described_df)

    features_df.to_csv(OUTPUT_DIR / "customer_features.csv", index=False)
    clustered_df.to_csv(OUTPUT_DIR / "customer_clusters.csv", index=False)
    cluster_profile_df.to_csv(OUTPUT_DIR / "cluster_profile.csv")
    final_cluster_book.to_csv(OUTPUT_DIR / "cluster_book_with_macro_segments.csv")

    print("Pipeline completed successfully.")
    print("Model metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
