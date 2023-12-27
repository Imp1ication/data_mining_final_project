import numpy as np
import time
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count


@dataclass
class KMeansResult:
    centers: np.ndarray
    cluster_result: np.ndarray
    sse: float = 0.0
    avg_sse: float = 0.0
    run_time: float = 0.0
    iter_count: int = 0


class KMeans:
    def __init__(
        self,
        n_clusters=5,
        max_iter=300,
        n_init=1,
        init_method="random",
        algorithm="lloyd",
        tolerance=1e-4,
        random_state=None,
        n_jobs=1,  # 0 means using all processors
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_method = init_method
        self.algorithm = algorithm
        self.tolerance = tolerance
        self.random_state = random_state
        self.n_jobs = n_jobs if n_jobs != 0 else cpu_count()

    # -- Initialize centers -- #
    def _init_centers(self, data):
        data_array = np.asarray(data)

        if self.init_method == "random":
            np.random.seed(self.random_state)
            return data_array[
                np.random.choice(
                    len(data_array),
                    size=self.n_clusters,
                    replace=False,
                )
            ]

        elif self.init_method == "kmeans++":
            return self._kmeans_plusplus_init(data_array)

        else:
            raise ValueError(
                "Invalid initialization method. Use 'random' or 'kmeans++'."
            )

    def _kmeans_plusplus_init(self, data):
        # Initialize first center
        np.random.seed(self.random_state)
        centers = [data[np.random.choice(len(data))]]

        # Initialize remaining centers
        for _ in range(1, self.n_clusters):
            # Compute distances from each point to nearest center
            distances = np.array(
                [min(np.linalg.norm(c - x) ** 2 for c in centers) for x in data]
            )

            probabilities = distances / distances.sum()
            new_center = data[np.random.choice(len(data), p=probabilities)]

            centers.append(new_center)

        return np.array(centers)

    # -- Assign points to closest center -- #
    def _assign_with_lloyd(self, data, centers):
        distances = np.array(
            [
                [np.linalg.norm(data_point - center) ** 2 for center in centers]
                for data_point in data
            ]
        )
        cluster_result = np.argmin(distances, axis=1)
        return cluster_result

    def _assign_with_elkan(self, data, centers):
        # Compute upper and lower bounds for each point
        lower_bounds = np.min(
            np.linalg.norm(data[:, np.newaxis, :] - centers, axis=2), axis=1
        )
        upper_bounds = np.min(
            np.linalg.norm(data[:, np.newaxis, :] + centers, axis=2), axis=1
        )

        distances = np.linalg.norm(data[:, np.newaxis, :] - centers, axis=2)
        cluster_result = np.argmin(distances, axis=1)

        # Refine assignments using Elkan's optimization
        for i in range(data.shape[0]):
            if upper_bounds[i] < lower_bounds[i]:
                distances_i = np.linalg.norm(data[i] - centers, axis=1)
                cluster_result[i] = np.argmin(distances_i)

        return cluster_result

    # def _assign_to_closest_center(self, data, centers):
    #     if self.algorithm == "lloyd":
    #         return self._assign_with_lloyd(data, centers)
    #     elif self.algorithm == "elkan":
    #         return self._assign_with_elkan(data, centers)
    #     else:
    #         raise ValueError("Invalid algorithm. Use 'lloyd' or 'elkan'.")

    def _assign_to_closest_parallel(self, data, centers, func):
        pool_sz = self.n_jobs
        data_chunks = [
            data[idx] for idx in np.array_split(np.arange(len(data)), pool_sz)
        ]

        with Pool(pool_sz) as p:
            results = p.starmap(func, [(chunk, centers) for chunk in data_chunks])

        return np.concatenate(results)

    def _assign_to_closest_center(self, data, centers):
        if self.algorithm == "lloyd":
            return self._assign_to_closest_parallel(
                data, centers, self._assign_with_lloyd
            )
        elif self.algorithm == "elkan":
            return self._assign_to_closest_parallel(
                data, centers, self._assign_with_elkan
            )
        else:
            raise ValueError("Invalid algorithm. Use 'lloyd' or 'elkan'.")

    # -- Update centers -- #
    def _update_cluster_centers_parallel(self, data, centers, cluster_result):
        updated_centers = []

        for i in range(self.n_clusters):
            cluster_i_data = data[cluster_result == i]

            if np.sum(cluster_result == i) > 0:
                new_center = np.mean(cluster_i_data, axis=0)
            else:
                new_center = centers[i]

            updated_centers.append(new_center)

        return np.array(updated_centers)

    def _update_cluster_centers(self, data, centers, cluster_result):
        pool_sz = self.n_jobs
        data_chunks = [
            data[idx] for idx in np.array_split(np.arange(len(data)), pool_sz)
        ]
        clus_chunks = [
            cluster_result[idx]
            for idx in np.array_split(np.arange(len(cluster_result)), pool_sz)
        ]

        with Pool(pool_sz) as p:
            results = p.starmap(
                self._update_cluster_centers_parallel,
                [
                    (data_c, centers, clus_c)
                    for data_c, clus_c in zip(data_chunks, clus_chunks)
                ],
            )

        return np.concatenate(results)

    # -- Calculate SSE -- #
    def _calc_sse(self, data, centers, cluster_result):
        sse = 0

        for i, center_index in enumerate(cluster_result):
            center = centers[center_index]
            dist_squared = np.sum((data[i] - center) ** 2)

            sse += dist_squared

        return sse

    # -- KMeans -- #
    def kmeans_once(self, data):
        start_time = time.time()

        centers = self._init_centers(data)
        cluster_result = self._assign_to_closest_center(data, centers)
        sse = self._calc_sse(data, centers, cluster_result)

        result = KMeansResult(
            centers=centers,
            cluster_result=cluster_result,
            sse=sse,
            iter_count=0,
        )

        for result.iter_count in range(self.max_iter):
            # Update centers
            new_centers = self._update_cluster_centers_parallel(
                data, result.centers, result.cluster_result
            )
            new_cluster_result = self._assign_to_closest_center(data, new_centers)

            new_sse = self._calc_sse(data, new_centers, new_cluster_result)

            if np.linalg.norm(new_centers - result.centers) < self.tolerance:
                break

            result.centers = new_centers
            result.cluster_result = new_cluster_result
            result.sse = new_sse

        result.run_time = time.time() - start_time
        return result

    def fit(self, data):
        data = np.asarray(data)

        best_result = KMeansResult(
            centers=np.zeros((self.n_clusters, data.shape[1])),
            cluster_result=np.zeros(data.shape[0]),
        )
        best_sse = np.inf
        total_sse = 0.0
        total_run_time = 0.0

        for _ in range(self.n_init):
            result = self.kmeans_once(data)

            total_sse += result.sse
            total_run_time += result.run_time

            if result.sse < best_sse:
                best_result = result
                best_sse = result.sse

        best_result.avg_sse = total_sse / self.n_init
        best_result.run_time = total_run_time

        return best_result
