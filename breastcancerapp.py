
"""
streamlit_app.py

Streamlit Web App – Advanced Breast Cancer Classification with PCA & Trained Model
"""
python -c "import matplotlib.pyplot as plt; import seaborn as sns"

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    import sys
    sys.exit("matplotlib is not installed! Check your requirements.txt")


# ----------------------------
# Configuration
# ----------------------------
st.set_page_config(page_title="Breast Cancer PCA & ML Pipeline", layout="wide")
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return data, df

data, df = load_data()
X = df.iloc[:, :-1]
y = df["target"]
target_names = data.target_names

# ----------------------------
# Sidebar - User Settings
# ----------------------------
st.sidebar.header("Settings")
show_data = st.sidebar.checkbox("Show Dataset", value=True)
show_summary = st.sidebar.checkbox("Show Summary Statistics", value=True)
show_class_dist = st.sidebar.checkbox("Show Class Distribution", value=True)

# List trained model files
model_files = [f for f in [
    "best_model_Logistic_Regression.pkl",
    "best_model_Random_Forest.pkl",
    "best_model_SVM_(RBF).pkl"
] if os.path.exists(f"{OUT_DIR}/{f}")]
best_model_file = st.sidebar.selectbox("Select Trained Model", model_files)

# ----------------------------
# Data Exploration
# ----------------------------
st.title("🔬 Breast Cancer Classification – PCA & ML Pipeline")

if show_data:
    st.subheader("Dataset – First 5 Rows")
    st.dataframe(df.head())

if show_summary:
    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

if show_class_dist:
    st.subheader("Class Distribution")
    vc = y.value_counts()
    st.bar_chart(vc)
    if vc.min() / vc.max() < 0.9:
        st.warning("⚠️ Classes are imbalanced.")
    else:
        st.success("✔ Classes are balanced.")

# ----------------------------
# Preprocessing & PCA
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_95 = PCA(n_components=0.95)
X_pca = pca_95.fit_transform(X_scaled)
n_components_95 = pca_95.n_components_

st.subheader("PCA Analysis")
st.write(f"Number of components for 95% variance: **{n_components_95}**")

# Explained variance plots
explained_ratio = pca_95.explained_variance_ratio_
cum_explained = np.cumsum(explained_ratio)

fig1, ax1 = plt.subplots(figsize=(8,4))
ax1.bar(range(1, min(11,len(explained_ratio))+1), explained_ratio[:10])
ax1.set_xlabel("PCA Component")
ax1.set_ylabel("Explained Variance Ratio")
ax1.set_title("Explained Variance Ratio (first 10 components)")
st.pyplot(fig1)

fig2, ax2 = plt.subplots(figsize=(8,4))
ax2.plot(range(1,len(cum_explained)+1), cum_explained, marker='o')
ax2.set_xlabel("Number of Components")
ax2.set_ylabel("Cumulative Explained Variance")
ax2.set_title("Cumulative Explained Variance")
ax2.grid(True)
st.pyplot(fig2)

# ----------------------------
# Load Trained Model
# ----------------------------
st.subheader("Trained Model Evaluation")
best_model_path = f"{OUT_DIR}/{best_model_file}"
best_model = joblib.load(best_model_path)
st.write(f"Loaded trained model: **{best_model_file.replace('best_model_','').replace('.pkl','')}**")

# Predict on full dataset
X_train_pca = X_pca
y_train = y.values
y_pred = best_model.predict(X_train_pca)
acc = accuracy_score(y_train, y_pred)
cm = confusion_matrix(y_train, y_pred)

st.write(f"**Accuracy on full dataset:** `{acc:.4f}`")

st.subheader("Confusion Matrix")
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names, ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
st.pyplot(fig_cm)

st.subheader("Classification Report")
cr_df = pd.DataFrame(classification_report(y_train, y_pred, target_names=target_names, output_dict=True)).transpose()
st.dataframe(cr_df)

# ----------------------------
# 2D PCA Visualization + Decision Boundary
# ----------------------------
st.subheader("PCA 2D Visualization & Decision Boundary")

pca2 = PCA(n_components=2)
X_2d = pca2.fit_transform(X_scaled)

best_model.fit(X_2d, y_train)

# Scatter plot
fig_scatter, ax_scatter = plt.subplots(figsize=(8,6))
for label in np.unique(y_train):
    idx = np.where(y_train==label)
    ax_scatter.scatter(X_2d[idx,0], X_2d[idx,1], label=target_names[label], edgecolors='k', alpha=0.8)
ax_scatter.set_xlabel("PC1")
ax_scatter.set_ylabel("PC2")
ax_scatter.set_title("PCA 2D Scatter Plot")
ax_scatter.legend()
st.pyplot(fig_scatter)

# Decision boundary
x_min, x_max = X_2d[:,0].min()-1, X_2d[:,0].max()+1
y_min, y_max = X_2d[:,1].min()-1, X_2d[:,1].max()+1
xx, yy = np.meshgrid(np.linspace(x_min,x_max,300), np.linspace(y_min,y_max,300))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = best_model.predict(grid).reshape(xx.shape)

fig_db, ax_db = plt.subplots(figsize=(9,7))
ax_db.contourf(xx, yy, Z, alpha=0.25)
for label in np.unique(y_train):
    idx = np.where(y_train==label)
    ax_db.scatter(X_2d[idx,0], X_2d[idx,1], label=target_names[label], edgecolors='k', alpha=0.9)
ax_db.set_xlabel("PC1")
ax_db.set_ylabel("PC2")
ax_db.set_title("Decision Boundary of Trained Model")
ax_db.legend()
st.pyplot(fig_db)

st.success("🎉 Interactive PCA & ML Pipeline Complete!")




