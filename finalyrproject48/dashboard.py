import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Database Column Optimization System")

# Load dataset
try:
    df = pd.read_csv("optimized_dataset.csv")  # Make sure this CSV exists
    st.subheader("Dataset Overview")
    st.write("Dataset Shape:", df.shape)

    st.write("Columns in Dataset:")
    st.write(df.columns)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    if not numeric_df.empty:
        fig, ax = plt.subplots()
        sns.heatmap(numeric_df.corr(), cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.write("No numeric columns to display correlation heatmap.")

    # Feature importance file (if saved earlier)
    try:
        importance = pd.read_csv("feature_importance.csv")
        st.subheader("Top Important Features")
        st.dataframe(importance.head(10))
    except FileNotFoundError:
        st.write("Feature importance file not found.")

    st.success("Dataset optimization completed successfully.")

except FileNotFoundError:
    st.error("Dataset file 'optimized_dataset.csv' not found. Please check the file path.")





st.title("Database Column Optimization Dashboard")

# Load optimized dataset
df = pd.read_csv("optimized_dataset.csv")

st.subheader("Dataset Overview")
st.write("Rows and Columns:", df.shape)
st.write("Columns:", df.columns)

# -------- Correlation Heatmap --------
st.subheader("Correlation Heatmap")

numeric_df = df.select_dtypes(include=['int64','float64'])

fig1, ax1 = plt.subplots(figsize=(10,6))
sns.heatmap(numeric_df.corr(), cmap="coolwarm", ax=ax1)

st.pyplot(fig1)

