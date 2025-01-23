from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# User uploads data
data = load_user_data()

# User selects number of components
num_components = user_input("Select number of components", default=2)

# Perform PCA
pca = PCA(n_components=num_components)
reduced_data = pca.fit_transform(data)

# Visualize reduced dimensions
plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
plt.title("PCA Dimensionality Reduction")
st.pyplot(plt)
