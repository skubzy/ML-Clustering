import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Dataset Selection
# Loading the 'wine-quality' dataset from Hugging Face
print("Loading dataset...")
red_wine = load_dataset("codesignal/wine-quality", split='red').to_pandas()
white_wine = load_dataset("codesignal/wine-quality", split='white').to_pandas()
dataset = pd.concat([red_wine, white_wine], ignore_index=True)
print("Dataset loaded successfully!")
# 2. Preprocessing
# Dropping non-numeric or target columns to keep it unsupervised
X = dataset.drop(columns=['quality', 'type'], errors='ignore')  # Assuming 'quality' is the target and 'type' is non-numeric
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Clustering Analysis: Elbow Method
inertia = []
K = range(1, 11)
for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K, inertia, 'bx-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()

# Based on Elbow, let's pick k=3
k_optimal = 3
kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df['cluster_label'] = clusters

# 4. Visualization (PCA to 2D)
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=components[:, 0], y=components[:, 1], hue=clusters, palette='viridis')
plt.title(f'Cluster Visualization (k={k_optimal})')
plt.show()

# 5. Cluster Prediction (Supervised Learning)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, clusters, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# 6. Evaluation
print("\nSupervised Model Evaluation:")
print(classification_report(y_test, y_pred))