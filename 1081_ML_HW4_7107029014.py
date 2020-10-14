import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score, completeness_score
from sklearn.decomposition import PCA

# Load the faces datasets(need to update sklearn to the new version)
faces_X, faces_Y = fetch_olivetti_faces(return_X_y=True)

# PCA
face_X_pca = PCA(n_components=3).fit_transform(faces_X)

# ==================== KMeans ====================
silhouette_avgs = []
ks = range(2, 11)
for k in ks:
    kmeans_fit = KMeans(n_clusters=k).fit(face_X_pca)
    cluster_labels = kmeans_fit.labels_
    silhouette_avg = silhouette_score(face_X_pca, cluster_labels)
    silhouette_avgs.append(silhouette_avg)

# 作圖並印出 k = 2 到 10 的績效
plt.bar(ks, silhouette_avgs)
plt.show()
print("clusters= 2~10 之績效: ")
print(silhouette_avgs)

# 由圖可知 n_clusters = 2 時之績效最好
print("=" * 20 + " KMeans " + "=" * 20)
kmeans_fit = KMeans(n_clusters=2)
Yk = kmeans_fit.fit_predict(face_X_pca)

print("Completeness score: ")
print(completeness_score(faces_Y, Yk))  # 0.763547062860643

# ==================== AgglomerativeClustering ====================

# 與KMeans 相同 n_clusters = 2 時之績效最好
print("=" * 20 + " Agglomerative Clustering " + "=" * 20)
ac = AgglomerativeClustering(n_clusters=2, linkage='ward')
Ya = ac.fit_predict(faces_X)

print("Completeness score: ")
print(completeness_score(faces_Y, Ya))  # 0.9080145866317462

# ==================== SpectralClustering ====================
print("=" * 20 + " Spectral Clustering " + "=" * 20)

# 與KMeans 相同 n_clusters = 2 時之績效最好
sc = SpectralClustering(n_clusters=2)
Yss = sc.fit_predict(face_X_pca)

print("Completeness score: ")
print(completeness_score(faces_Y, Yss))  # 0.6995565948493547
