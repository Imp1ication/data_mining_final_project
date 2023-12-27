from scripts import preprocess
from scripts import kmeans

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler


def run_experiment_rvk(init_method, df_for_clustering, n_clusters_range):
    sse_values = []
    avg_sse_values = []
    time_values = []

    for n_cluster in n_clusters_range:
        print(f"n_cluster: {n_cluster}")

        kmeans_instance = kmeans.KMeans(
            n_clusters=n_cluster,
            n_init=5,
            init_method=init_method,
            tolerance=1e-7,
            random_state=None,
            n_jobs=0,
        )

        result = kmeans_instance.fit(df_for_clustering)
        sse_values.append(result.sse)
        avg_sse_values.append(result.avg_sse)
        time_values.append(result.run_time)

        print(f"{init_method}_sse: {result.sse}")
        print(f"{init_method}_avg_sse: {result.avg_sse}")
        print(f"{init_method}_time: {result.run_time}\n")

    return sse_values, avg_sse_values, time_values


def run_experiment_lve(algorithm, df_for_clustering, n_clusters_range):
    sse_values = []
    time_values = []

    for n_cluster in n_clusters_range:
        print(f"n_cluster: {n_cluster}")

        kmeans_instance = kmeans.KMeans(
            n_clusters=n_cluster,
            init_method="kmeans++",
            algorithm=algorithm,
            random_state=17,
            n_jobs=0,
        )

        result = kmeans_instance.fit(df_for_clustering)
        sse_values.append(result.sse)
        time_values.append(result.run_time)

        print(f"{algorithm}_sse: {result.sse}")
        print(f"{algorithm}_time: {result.run_time}\n")

    return sse_values, time_values


def plot_results(n_clusters_range, values, title, label, ylabel, filename):
    plt.figure(figsize=(10, 6))

    for method, value in values.items():
        plt.plot(n_clusters_range, value, marker="o", label=method)

    plt.title(title)
    plt.xlabel(label)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(f"outputs/kmeans_analyze/{filename}")
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

    df_for_clustering = df.drop(["On_chart"], axis=1)

    # -- Random vs kmeans++ -- #
    print(">-- Random vs KMeans++ --<")

    n_clusters_range = range(2, 16)

    random_sse, random_avg_sse, random_time = run_experiment_rvk(
        "random",
        df_for_clustering,
        n_clusters_range,
    )
    kpp_sse, kpp_avg_sse, kpp_time = run_experiment_rvk(
        "kmeans++",
        df_for_clustering,
        n_clusters_range,
    )

    plot_results(
        n_clusters_range,
        {"Randon Initialization": random_sse, "KMeans++ Initialization": kpp_sse},
        "KMeans SSE for Different Initialization Methods",
        "Number of Clusters (k)",
        "SSE (Sum of Squared Errors)",
        "rvk_5_sse.png",
    )

    plot_results(
        n_clusters_range,
        {
            "Randon Initialization": random_avg_sse,
            "KMeans++ Initialization": kpp_avg_sse,
        },
        "KMeans Average SSE for Different Initialization Methods",
        "Number of Clusters (k)",
        "Average SSE (Sum of Squared Errors)",
        "rvk_5_avg_sse.png",
    )

    plot_results(
        n_clusters_range,
        {
            "Randon Initialization": random_time,
            "KMeans++ Initialization": kpp_time,
        },
        "KMeans Runtime for Different Initialization Methods",
        "Number of Clusters (k)",
        "Runtime (seconds)",
        "rvk_5_run_time.png",
    )

    # -- Lloyd vs Elkan -- #
    print(">-- Lloyd vs Elkan --<")

    n_clusters_range = range(2, 11)

    lloyd_sse, lloyd_time = run_experiment_lve(
        "lloyd",
        df_for_clustering,
        n_clusters_range,
    )

    elkan_sse, elkan_time = run_experiment_lve(
        "elkan",
        df_for_clustering,
        n_clusters_range,
    )

    plot_results(
        n_clusters_range,
        {
            "Lloyd": lloyd_sse,
            "Elkan": elkan_sse,
        },
        "KMeans SSE for Different Algorithm",
        "Number of Clusters (k)",
        "SSE (Sum of Squared Errors)",
        "lve_sse.png",
    )

    plot_results(
        n_clusters_range,
        {
            "Lloyd": lloyd_time,
            "Elkan": elkan_time,
        },
        "KMeans Runtime for Different Algorithm",
        "Number of Clusters (k)",
        "Runtime (seconds)",
        "lve_run_time.png",
    )
