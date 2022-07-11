import pandas as pd
import plotly.express as px
df = pd.read_csv('stars.csv')
print(df.head())
fig = px.scatter(df, x = 'Size', y = 'Light')
fig.show()
from sklearn.cluster import KMeans
X = df.iloc[:, [0, 1]].values
print(X)
WCSS = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    WCSS.append(kmeans.inertia_)
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize = (10, 5))
sns.lineplot(range(1, 11), WCSS, marker = 'o', color = 'red')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
yKmeans = kmeans.fit_predict(X)
plt.figure(figsize = (15, 7))
sns.scatterplot(X[yKmeans == 0, 0], X[yKmeans == 0, 1], color = 'yellow', label = 'Cluster 1')
sns.scatterplot(X[yKmeans == 1, 0], X[yKmeans == 1, 1], color = 'blue', label = 'Cluster 2')
sns.scatterplot(X[yKmeans == 2, 0], X[yKmeans == 2, 1], color = 'green', label = 'Cluster 3')
sns.scatterplot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color = 'red', label = 'Centroids', s = 100, marker = ',')
plt.grid(False)
plt.title('Clusters of Interstellar Objects')
plt.xlabel('Size')
plt.ylabel('Light')
plt.legend()
plt.show()