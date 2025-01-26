import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Function to generate random data (for demonstration)
def generate_data(n_samples=300, n_features=5, n_clusters=3):
    np.random.seed(42)
    X = np.random.rand(n_samples, n_features)
    return pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(n_features)])

# Reusable plotting function
def plot_results(data, x_col, y_col, label_col=None, title="Plot"):
    """
    Plots the 2D data and optionally color-codes it based on labels.

    Args:
        data (pd.DataFrame): DataFrame containing the columns to plot.
        x_col (str): Name of the column for the x-axis.
        y_col (str): Name of the column for the y-axis.
        label_col (str): Optional; Name of the column with labels for color-coding.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(8, 6))
    if label_col:
        sns.scatterplot(data=data, x=x_col, y=y_col, hue=label_col, palette="viridis", s=50)
        plt.legend(title=label_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        sns.scatterplot(data=data, x=x_col, y=y_col, s=50, color="blue")
    plt.title(title, fontsize=14)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# Main: Generate data, perform clustering, and dimensionality reduction
data = generate_data()

# 1. Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data["Cluster"] = kmeans.fit_predict(data)

# 2. Dimensionality Reduction
pca = PCA(n_components=2)
data[["PC1", "PC2"]] = pca.fit_transform(data)

# 3. Plot Clustering Results
plot_results(data, x_col="PC1", y_col="PC2", label_col="Cluster", title="Clustering Results (K-Means)")

# 4. Plot Dimensionality Reduction Results
# (Here, no labels are passed, as we are just visualizing the PCA result)
plot_results(data, x_col="PC1", y_col="PC2", title="Dimensionality Reduction (PCA)")
