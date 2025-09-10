import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from clustering import run_gmm
from visualizations import (
    gmm_simple_plot,
    membership_table,
    cluster_distribution_chart,
    avg_spend_per_segment,
    revenue_contribution_chart,
    high_value_pie,
    high_value_scatter
)

st.set_page_config(page_title="GMM Audience Segmentation", layout="wide")
st.title("Audience Segmentation using GMM (Soft Clustering)")
st.markdown("### 📌 Required Columns in Your CSV")
required_cols_info = [
    "User_ID",
    "Total_Spend",
    "Purchase_Frequency",
    "Avg_Order_Value",
    "Last_Purchase_Days_Ago",
    "Loyalty_Score",
    "Impressions",
    "Clicks",
    "Spent",
    "Total_Conversion",
    "Approved_Conversion"
]
st.write(required_cols_info)
st.info("Make sure your uploaded CSV has all these columns for successful segmentation.")

st.sidebar.header("Upload User Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV with user data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Raw Uploaded Data", df.head())

    required_cols = ["User_ID", "Total_Spend", "Purchase_Frequency"]
    if not all(col in df.columns for col in required_cols):
        st.error(f"Missing columns! Required: {required_cols}")
    else:
        st.success("Data validated successfully!")

        # --- Data Preprocessing ---
        st.subheader("Step 1: Data Cleaning & Preprocessing")
        df_clean = df.dropna(subset=required_cols).copy()
        features = df_clean[["Total_Spend", "Purchase_Frequency"]]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)

        st.write("Cleaned Data Shape:", df_clean.shape)

        # --- GMM Clustering ---
        st.subheader("Step 2: Apply GMM Clustering")
        n_clusters = st.slider("Select number of clusters", min_value=2, max_value=5, value=3)
        labels, probs, gmm_model = run_gmm(X_scaled, n_components=n_clusters)

        cluster_mapping = {
            0: "High Spender",
            1: "Loyal Customer",
            2: "Window Shopper"
        }

        df_clean["Cluster"] = labels
        df_clean["Predicted Segment"] = df_clean["Cluster"].map(cluster_mapping)

        # --- Visualization ---
        st.subheader("Step 3: Visualize Clusters")
        fig = gmm_simple_plot(
            X=features,
            y_pred=labels,
            gmm_model=gmm_model,
            feature_x="Total_Spend",
            feature_y="Purchase_Frequency",
            cluster_mapping=cluster_mapping
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Probability Table ---
        st.subheader("Step 4: Soft Membership Probabilities")
        prob_df = membership_table(
            user_ids=df_clean["User_ID"],
            gmm_probs=probs,
            cluster_labels=labels,
            cluster_mapping=cluster_mapping
        )
        st.dataframe(prob_df)

        csv_download = prob_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Segmentation Results",
            data=csv_download,
            file_name="gmm_user_segmentation.csv",
            mime="text/csv"
        )

        # --- Cluster Insights ---
        st.subheader("Step 5: Cluster Insights")

        st.write("### 1. User Distribution per Segment")
        st.plotly_chart(cluster_distribution_chart(df_clean), use_container_width=True)

        st.write("### 2. Average Spend per Segment")
        st.plotly_chart(avg_spend_per_segment(df_clean), use_container_width=True)

        # --- New Revenue-Focused Insights ---
        st.subheader("Step 6: Revenue-Focused Targeting Insights")

        st.write("### 3. Revenue Contribution by Segment")
        st.plotly_chart(revenue_contribution_chart(df_clean), use_container_width=True)

        st.write("### 4. High Value vs Low Value Users")
        st.plotly_chart(high_value_pie(df_clean), use_container_width=True)

        st.write("### 5. High Value Users Scatter")
        st.plotly_chart(high_value_scatter(df_clean), use_container_width=True)

else:
    st.info("Upload a CSV to begin segmentation.")
