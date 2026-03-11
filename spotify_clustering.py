# Step 1: Import libraries
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns


# Step 2: Load Dataset
df = pd.read_csv("SpotifyFeatures.csv")

# Display first rows
print(df.head())


# Step 3: Select Required Features
features = ['danceability', 'energy', 'tempo', 'loudness', 'valence']

X = df[features]

# Handle missing values: Drop rows with any NaN values in the selected features
X = X.dropna()

# Ensure df also reflects these dropped rows for later plotting
df = df.loc[X.index]


# Step 4: Normalize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Step 5: Apply KMeans Clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10) # Added n_init for newer sklearn versions
clusters = kmeans.fit_predict(X_scaled)

# Add cluster column to dataset
df['Cluster'] = clusters


# Step 6: PCA Visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

df['PCA1'] = pca_result[:,0]
df['PCA2'] = pca_result[:,1]


plt.figure(figsize=(8,6))
sns.scatterplot(
    x='PCA1',
    y='PCA2',
    hue='Cluster',
    palette='Set1',
    data=df
)

plt.title("Spotify Song Clusters using PCA")
plt.show()


# Step 7: t-SNE Visualization
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(X_scaled)

df['TSNE1'] = tsne_result[:,0]
df['TSNE2'] = tsne_result[:,1]


plt.figure(figsize=(8,6))
sns.scatterplot(
    x='TSNE1',
    y='TSNE2',
    hue='Cluster',
    palette='Set2',
    data=df
)

plt.title("Spotify Song Clusters using t-SNE")
plt.show()


# Step 8: Cluster Insights
cluster_summary = df.groupby('Cluster')[features].mean()

print("\nCluster Characteristics:")
print(cluster_summary)
