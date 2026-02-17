# =====================================
# Streamlit Dashboard: Trader Performance & Market Sentiment (Enhanced)
# =====================================

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt

# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(
    page_title="Trader Performance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
sns.set(style="whitegrid")

# ----------------------------
# Load data
# ----------------------------
@st.cache_data
def load_data():
    merged = pd.read_csv("final_merged_dataset.csv")
    trader_features = pd.read_csv("trader_clusters.csv")
    merged = merged.merge(trader_features[['account','cluster']], on='account', how='left')
    return merged, trader_features

merged, trader_features = load_data()

# ----------------------------
# Sidebar filters
# ----------------------------
st.sidebar.title("Filters")

sentiment_options = merged['classification'].unique()
selected_sentiments = st.sidebar.multiselect(
    "Select Market Sentiments:",
    options=sentiment_options,
    default=sentiment_options
)

cluster_options = trader_features['cluster'].unique()
selected_clusters = st.sidebar.multiselect(
    "Select Trader Clusters:",
    options=cluster_options,
    default=cluster_options
)

filtered_merged = merged[merged['classification'].isin(selected_sentiments)]
filtered_clusters = trader_features[trader_features['cluster'].isin(selected_clusters)]

# ----------------------------
# Dashboard Title & KPIs
# ----------------------------
st.title("📊 Trader Performance & Market Sentiment Dashboard ")

# KPIs
total_trades = filtered_merged.shape[0]
total_pnl = filtered_merged['closed_pnl'].sum()
overall_win_rate = filtered_merged['win'].mean() * 100

col1, col2, col3 = st.columns(3)
col1.metric("Total Trades", f"{total_trades:,}")
col2.metric("Total PnL (USD)", f"${total_pnl:,.0f}")
col3.metric("Overall Win Rate", f"{overall_win_rate:.2f}%")

# ----------------------------
# Dataset Overview
# ----------------------------
st.header("Dataset Overview")
st.markdown("**Merged Trades Dataset:**")
st.dataframe(filtered_merged.head(10))
st.markdown("**Trader Clusters Dataset:**")
st.dataframe(filtered_clusters.head(10))

# ----------------------------
# Market Sentiment Analysis
# ----------------------------
st.header("Market Sentiment vs Trader Performance")

# PnL Distribution
st.subheader("PnL Distribution by Market Sentiment")
fig, ax = plt.subplots(figsize=(10,5))
sns.boxplot(
    x='classification', 
    y='closed_pnl', 
    data=filtered_merged, 
    palette="coolwarm",
    ax=ax
)
ax.set_xlabel("Market Sentiment")
ax.set_ylabel("Closed PnL (USD)")
ax.set_title("PnL Distribution by Market Sentiment")
plt.xticks(rotation=45)
st.pyplot(fig)

# Win Rate by sentiment
st.subheader("Win Rate by Market Sentiment (%)")
win_rate = filtered_merged.groupby('classification')['win'].mean() * 100
st.bar_chart(win_rate)

# ----------------------------
# Trade Side Performance
# ----------------------------
st.header("Trade Side Performance")
side_win = filtered_merged.groupby('side')['win'].mean() * 100
st.bar_chart(side_win)

# ----------------------------
# Trade Size vs Win Rate
# ----------------------------
st.header("Trade Size vs Win Rate")
filtered_merged['size_bin'] = pd.qcut(filtered_merged['size_usd'], 4, duplicates='drop')
size_win = filtered_merged.groupby('size_bin')['win'].mean() * 100
fig2, ax2 = plt.subplots(figsize=(10,5))
sns.barplot(x=size_win.index.astype(str), y=size_win.values, palette="viridis", ax=ax2)
ax2.set_xlabel("Trade Size Bin")
ax2.set_ylabel("Win Rate (%)")
ax2.set_title("Trade Size vs Win Rate")
plt.xticks(rotation=45)
st.pyplot(fig2)

# ----------------------------
# Trader Clustering Insights
# ----------------------------
st.header("Trader Clustering Insights")

# Win Rate by Cluster
st.subheader("Win Rate Distribution by Cluster")
fig3, ax3 = plt.subplots(figsize=(10,5))
sns.boxplot(
    x='cluster', y='win_rate', 
    data=filtered_clusters, palette="Set2", ax=ax3
)
ax3.set_xlabel("Cluster")
ax3.set_ylabel("Win Rate")
ax3.set_title("Win Rate Distribution by Cluster")
st.pyplot(fig3)

# Total PnL by Cluster
st.subheader("Total PnL Distribution by Cluster")
fig4, ax4 = plt.subplots(figsize=(10,5))
sns.boxplot(
    x='cluster', y='total_pnl', 
    data=filtered_clusters, palette="Set3", ax=ax4
)
ax4.set_xlabel("Cluster")
ax4.set_ylabel("Total PnL")
ax4.set_title("Total PnL Distribution by Cluster")
st.pyplot(fig4)

# Cluster Summary Metrics
st.subheader("Cluster Summary Metrics")
st.dataframe(filtered_clusters.groupby('cluster').mean(numeric_only=True))

# Top Traders per Cluster
st.header("🏆 Top Profitable Traders per Cluster")
top_traders = trader_features.groupby('cluster').apply(lambda x: x.nlargest(5,'total_pnl')).reset_index(drop=True)
st.dataframe(top_traders[['account','cluster','total_trades','win_rate','total_pnl']])

# ----------------------------
# Win Rate Heatmap: Cluster vs Sentiment
# ----------------------------
st.header("🔥 Win Rate Heatmap: Cluster vs Sentiment")
heatmap_data = merged.pivot_table(
    index='cluster', columns='classification', values='win', aggfunc='mean'
) * 100

fig5, ax5 = plt.subplots(figsize=(12,5))
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=.5, ax=ax5)
ax5.set_xlabel("Market Sentiment")
ax5.set_ylabel("Cluster")
ax5.set_title("Win Rate (%) by Cluster and Sentiment")
st.pyplot(fig5)

# ----------------------------
# Interactive Scatter Plot
# ----------------------------
st.header("📈 Trade Size vs PnL (Interactive)")
scatter = alt.Chart(filtered_merged).mark_circle(size=60).encode(
    x=alt.X('size_usd', title='Trade Size (USD)'),
    y=alt.Y('closed_pnl', title='Closed PnL'),
    color=alt.Color('cluster:N', title='Cluster'),
    tooltip=['account','side','size_usd','closed_pnl','cluster','win']
).interactive().properties(width=800, height=400)
st.altair_chart(scatter)

# ----------------------------
# Completion message
# ----------------------------
st.success("✅ Enhanced dashboard loaded! Use sidebar filters to explore traders, clusters, sentiment, trade size, and profitability interactively.")



