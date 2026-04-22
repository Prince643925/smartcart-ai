#  “This module handles data preprocessing including cleaning, feature engineering, encoding, scaling, and dimensionality reduction using PCA.”

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def preprocess_data(df):

    # ✅ Handle missing values safely
    if "AnnualIncome" in df.columns:
        df["AnnualIncome"] = df["AnnualIncome"].fillna(df["AnnualIncome"].median())

    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())

    if "SpendingScore" in df.columns:
        df["SpendingScore"] = df["SpendingScore"].fillna(df["SpendingScore"].median())

    # ✅ Select only numeric columns
    df_clean = df.select_dtypes(include=["int64", "float64"])

    # ❌ Drop CustomerID (not useful for clustering)
    if "CustomerID" in df_clean.columns:
        df_clean = df_clean.drop(columns=["CustomerID"])

    # ✅ Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)

    # ✅ PCA (reduce to 2D for clustering)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, df_clean