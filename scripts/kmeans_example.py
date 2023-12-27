import numpy as np
import kmeans  # 將 kmeans.py 放在同一個目錄下

# 範例數據
data = np.array([[1, 2], [5, 8], [1.5, 1.8], [8, 8], [1, 0.6], [9, 11]])

# 初始化類別
kmeans_cluster = kmeans.KMeans(
    n_clusters=3,
    n_init=5,
    init_method="kmeans++",
    algorithm="lloyd",
    random_state=17,
)

# 執行K-means clustering
result = kmeans_cluster.fit(data)

# 印出結果，四捨五入至小數點後第二位
print(f"資料集：\n{np.round(data, 2)}\n")
print(f"最佳聚類中心：\n{np.round(result.centers, 2)}\n")
print(f"最佳分配結果：\n{result.cluster_result}\n")
print(f"最佳SSE值： {round(result.sse, 2)}")
print(f"平均SSE值： {round(result.avg_sse, 2)}\n")
print(f"算法執行時間： {round(result.run_time, 2)} 秒")
