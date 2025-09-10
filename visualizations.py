import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.mixture import GaussianMixture

def gmm_simple_plot(X: pd.DataFrame, y_pred: np.ndarray, gmm_model: GaussianMixture,
                    feature_x: str, feature_y: str, cluster_mapping: dict) -> go.Figure:
    fig = go.Figure()

    # Scatter for each cluster
    for cluster in np.unique(y_pred):
        mask = y_pred == cluster
        fig.add_trace(go.Scatter(
            x=X[feature_x][mask],
            y=X[feature_y][mask],
            mode="markers",
            name=cluster_mapping.get(cluster, f"Cluster {cluster}"),
            marker=dict(size=8, opacity=0.7)
        ))

    # Add proper axis labels
    fig.update_layout(
        title=f"GMM Clustering ({feature_x} vs {feature_y})",
        xaxis_title=feature_x.replace("_", " "),
        yaxis_title=feature_y.replace("_", " "),
        template="plotly"
    )
    return fig


def membership_table(user_ids: pd.Series, gmm_probs: np.ndarray,
                     cluster_labels: np.ndarray, cluster_mapping: dict) -> pd.DataFrame:
    prob_df = pd.DataFrame(gmm_probs, columns=[f"Prob_{cluster_mapping.get(i, i)}"
                                               for i in range(gmm_probs.shape[1])])
    prob_df.insert(0, "UserID", user_ids.values)
    prob_df.insert(1, "Predicted Segment", [cluster_mapping.get(c, c) for c in cluster_labels])
    return prob_df

def cluster_distribution_chart(df: pd.DataFrame, segment_col: str = "Predicted Segment") -> go.Figure:
    segment_counts = df[segment_col].value_counts().reset_index()
    segment_counts.columns = ["Segment", "Count"]
    fig = go.Figure(go.Bar(
        x=segment_counts["Segment"],
        y=segment_counts["Count"],
        text=segment_counts["Count"],
        textposition='outside'
    ))
    return fig

def avg_spend_per_segment(df: pd.DataFrame,
                          spend_col: str = "Total_Spend",
                          segment_col: str = "Predicted Segment") -> go.Figure:
    avg_spend = df.groupby(segment_col)[spend_col].mean().reset_index()
    fig = go.Figure(go.Bar(
        x=avg_spend[segment_col],
        y=avg_spend[spend_col],
        text=avg_spend[spend_col].round(2),
        textposition='outside'
    ))
    return fig

# --- NEW GRAPHS ---
def revenue_contribution_chart(df: pd.DataFrame,
                               spend_col: str = "Total_Spend",
                               segment_col: str = "Predicted Segment") -> go.Figure:
    revenue_by_segment = df.groupby(segment_col)[spend_col].sum().reset_index()
    fig = px.bar(revenue_by_segment,
                 x=segment_col,
                 y=spend_col,
                 title="Revenue Contribution by Segment",
                 text=spend_col,
                 color=segment_col)
    return fig

def high_value_pie(df: pd.DataFrame,
                   segment_col: str = "Predicted Segment") -> go.Figure:
    df["Value_Group"] = df[segment_col].apply(
        lambda x: "High Value" if x in ["High Spender", "Loyal Customer"] else "Low Value"
    )
    counts = df["Value_Group"].value_counts().reset_index()
    counts.columns = ["Group", "Count"]
    fig = px.pie(counts, names="Group", values="Count",
                 title="High Value vs Low Value Users", hole=0.4)
    return fig

def high_value_scatter(df: pd.DataFrame,
                       spend_col: str = "Total_Spend",
                       freq_col: str = "Purchase_Frequency",
                       segment_col: str = "Predicted Segment") -> go.Figure:
    df["Value_Group"] = df[segment_col].apply(
        lambda x: "High Value" if x in ["High Spender", "Loyal Customer"] else "Low Value"
    )
    fig = px.scatter(
        df,
        x=spend_col,
        y=freq_col,
        color="Value_Group",
        size="Total_Spend",
        hover_data=[segment_col, spend_col, freq_col],
        title="High Value Users vs Others"
    )
    return fig
