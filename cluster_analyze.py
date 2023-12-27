from scripts import preprocess
from scripts import kmeans

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler


def cluster_analysis(data, title):
    print(f"Start analyzing {title}...")

    df_for_clustering = data.drop(["On_chart"], axis=1)

    kmeans_instance = kmeans.KMeans(
        n_clusters=10,
        n_init=10,
        init_method="kmeans++",
        algorithm="elkan",
        tolerance=1e-7,
        n_jobs=0,
    )

    kmeans_result = kmeans_instance.fit(df_for_clustering)

    cluster_column = pd.Series(kmeans_result.cluster_result, name="Cluster")
    clustered_data = pd.concat([data, cluster_column], axis=1)
    clustered_data.to_csv(f"outputs/{title}/{title}_clusters_result.csv", index=False)

    # Calculate cluster means
    cluster_means = clustered_data.groupby("Cluster").mean()
    cluster_means = pd.DataFrame(cluster_means).reset_index()
    cluster_means.drop(["On_chart"], axis=1, inplace=True)

    # Calculate on-chart ratio
    cluster_sizes = clustered_data.groupby("Cluster").size()
    on_chart_counts = (
        clustered_data[clustered_data["On_chart"] == 1].groupby("Cluster").size()
    )
    on_chart_ratio = on_chart_counts / cluster_sizes

    # Merge results
    result_df = pd.DataFrame(
        {
            "Cluster": on_chart_counts.index,
            "Cluster_Size": cluster_sizes.values,
            "On_chart_ratio": on_chart_ratio.values,
        }
    )

    result_df = pd.merge(result_df, cluster_means, on="Cluster")
    result_df = result_df.round(3)
    result_df.to_csv(f"outputs/{title}/{title}_analysis_result.csv", index=False)

    # plot On-chart ratio
    plt.figure(figsize=(10, 6))
    plt.bar(result_df["Cluster"], result_df["On_chart_ratio"])
    plt.title("On_chart Ratio in Each Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("On_chart Ratio")
    plt.savefig(f"outputs/{title}/{title}_on_chart_ratio.png")

    # plot correlation
    correlations = result_df.iloc[:, 3:].corrwith(result_df["On_chart_ratio"])
    # correlations = correlations.sort_values(ascending=False)

    plt.figure(figsize=(12, 8))
    correlations.plot(kind="bar")
    plt.title("Correlation between Features and On_chart_ratio")
    plt.xlabel("Features")
    plt.xticks(rotation=30)
    plt.ylabel("Correlation Coefficient")
    plt.savefig(f"outputs/{title}/{title}_on_chart_correlations.png")
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("data.csv")

    # -- Preprocessing -- #
    # Missing value
    preprocess.check_missing_values(
        df, output_csv_path="outputs/missing_values_report.csv"
    )

    df = df.drop(["track_title", "artist_name", "track_id"], axis=1)

    # MinMaxScaling
    min_max_scaler = MinMaxScaler()
    min_max_features = [
        "duration_ms",
        "tempo",
        "key",
        "mode",
        "time_signature",
        "loudness",
    ]

    df[min_max_features] = min_max_scaler.fit_transform(df[min_max_features])

    # -- Cluster Analysis -- #
    structure_features = [
        "duration_ms",
        "time_signature",
        "tempo",
        "key",
        "mode",
        "loudness",
        "On_chart",
    ]

    music_features = [
        "energy",
        "danceability",
        "valence",
        "liveness",
        "instrumentalness",
        "acousticness",
        "speechiness",
        "On_chart",
    ]

    cluster_analysis(df[structure_features], "structure_features")
    cluster_analysis(df[music_features], "music_features")
    cluster_analysis(df, "all_features")
