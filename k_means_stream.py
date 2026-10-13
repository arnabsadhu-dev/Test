import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

st.set_page_config(page_title="K-Means Clustering App", layout="wide")

st.title("📊 K-Means Clustering App")
st.write("Upload a CSV file or use the built-in Iris dataset to perform K-Means clustering.")

# Sidebar - Data Loading
st.sidebar.header("1. Load Data")
data_option = st.sidebar.radio("Select Dataset", ("Iris Dataset (Default)", "Upload CSV"))

if data_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("Please upload a CSV file to continue.")
        st.stop()
else:
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Display dataset preview
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Feature Selection
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

if len(numeric_cols) < 2:
    st.error("The dataset needs at least 2 numerical columns for clustering.")
    st.stop()

st.sidebar.header("2. Clustering Settings")
selected_features = st.sidebar.multiselect(
    "Select Features for Clustering",
    numeric_cols,
    default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols
)

if len(selected_features) < 2:
    st.warning("Please select at least 2 features.")
    st.stop()

# Hyperparameters
n_clusters = st.sidebar.slider("Number of Clusters (k)", min_value=2, max_value=10, value=3)
scale_data = st.sidebar.checkbox("Scale Features (StandardScaler)", value=True)

# Prepare Data
X = df[selected_features].dropna()

if scale_data:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
else:
    X_scaled = X.values

# Run K-Means
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to dataframe
df_result = df.loc[X.index].copy()
df_result["Cluster"] = cluster_labels

# Results Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Cluster Visualization")
    fig, ax = plt.subplots(figsize=(8, 6))

    if len(selected_features) == 2:
        # Direct 2D Plot
        sns.scatterplot(
            x=X_scaled[:, 0],
            y=X_scaled[:, 1],
            hue=cluster_labels,
            palette="viridis",
            style=cluster_labels,
            s=80,
            ax=ax
        )
        # Plot Cluster Centers
        centers = kmeans.cluster_centers_
        ax.scatter(centers[:, 0], centers[:, 1], c="red", marker="X", s=200, label="Centroids")
        ax.set_xlabel(selected_features[0] + (" (Scaled)" if scale_data else ""))
        ax.set_ylabel(selected_features[1] + (" (Scaled)" if scale_data else ""))
    else:
        # PCA for multi-feature visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        centers_pca = pca.transform(kmeans.cluster_centers_)

        sns.scatterplot(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            hue=cluster_labels,
            palette="viridis",
            style=cluster_labels,
            s=80,
            ax=ax
        )
        ax.scatter(centers_pca[:, 0], centers_pca[:, 1], c="red", marker="X", s=200, label="Centroids")
        ax.set_xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]:.1%} Var)")
        ax.set_ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]:.1%} Var)")

    ax.set_title(f"K-Means Clustering (k={n_clusters})")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("Cluster Metrics")
    st.metric("Inertia (WCSS)", f"{kmeans.inertia_:.2f}")

    st.subheader("Cluster Counts")
    st.write(df_result["Cluster"].value_counts().rename_axis("Cluster").reset_index(name="Count"))

# Data Table & Download
st.subheader("Clustered Dataset")
st.dataframe(df_result)

csv_data = df_result.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Clustered Data as CSV",
    data=csv_data,
    file_name="kmeans_clustered_data.csv",
    mime="text/csv"
)