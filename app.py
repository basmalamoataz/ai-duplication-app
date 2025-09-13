import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import nltk
import io
import re

nltk.download('punkt', quiet=True)

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="AI Duplication",
    page_icon="logo.png", # Updated to use the local image file
    layout="wide"
)

# -------------------------------
# Inject custom CSS
# -------------------------------
CUSTOM_CSS = """
<style>
/* (Your custom CSS is here - no changes needed) */
:root{--primary:#3B82F6;--bg:#0b1220;--panel:#121a2b;--text:#E5E7EB;--muted:#9CA3AF;--radius:14px;}
html, body, [class^="stApp"] {background: var(--bg); color: var(--text);}
.stApp > header {background: transparent;}
.stButton > button {background: linear-gradient(135deg, var(--primary), #2563EB); color: white; border: 0; border-radius: 10px; padding: 0.7rem 1.1rem; font-weight: 600;}
div[data-testid="stFileUploader"] {padding: 1rem; background: var(--panel); border: 1px solid rgba(255,255,255,0.06); border-radius: 14px;}
div[data-testid="stFileUploader"] section {background: rgba(0,0,0,0.2); border: 2px dashed var(--muted); border-radius: 14px;}
div[data-testid="stDataFrame"] {background: var(--panel); border: 1px solid rgba(255,255,255,0.06); border-radius: 14px;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -------------------------------
# Model cache
# -------------------------------
@st.cache_resource
def load_model():
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    return model

# -------------------------------
# Analysis Function
# -------------------------------
def find_similar_work_order_pairs(df, model):
    # (Your full analysis function is here - no changes needed)
    desc_column = "Description"
    wo_number_column = "Work Order Number"
    status_column = "Status"
    location_column = "Location"
    asset_column = "Asset"
    pm_number_column = "PM Number"
    statuses_to_exclude = ["COMP", "CAN"]
    similarity_threshold = 0.75
    if status_column in df.columns: df[status_column] = df[status_column].astype(str)
    filtered_df = df
    if status_column in df.columns: filtered_df = df[~df[status_column].isin(statuses_to_exclude)]
    if pm_number_column in filtered_df.columns:
        filtered_df[pm_number_column] = filtered_df[pm_number_column].astype(str)
        filtered_df = filtered_df[filtered_df[pm_number_column].isnull() | (filtered_df[pm_number_column] == 'nan')].copy()
    results_data = []
    if not {location_column, asset_column}.issubset(filtered_df.columns):
        st.error("Missing required columns for grouping: Location and/or Asset.")
        return pd.DataFrame()
    grouped = filtered_df.groupby([location_column, asset_column])
    for (location, asset), group_df in grouped:
        if len(group_df) < 2: continue
        sentences_df = group_df.dropna(subset=[desc_column])
        if len(sentences_df) < 2: continue
        sentences = sentences_df[desc_column].astype(str).tolist()
        work_orders = sentences_df[wo_number_column].tolist()
        embeddings = model.encode(sentences, show_progress_bar=False, convert_to_tensor=True)
        clusters = util.community_detection(embeddings, threshold=similarity_threshold, min_community_size=2)
        if clusters:
            for cluster in clusters:
                for i_pos in range(len(cluster)):
                    for j_pos in range(i_pos + 1, len(cluster)):
                        idx1, idx2 = cluster[i_pos], cluster[j_pos]
                        score = util.cos_sim(embeddings[idx1], embeddings[idx2])[0, 0].item()
                        results_data.append({"Similarity Score": f"{score:.4f}", "Work Order 1": work_orders[idx1], "Description 1": sentences[idx1], "Work Order 2": work_orders[idx2], "Description 2": sentences[idx2], "Location": location, "Asset": asset})
    if results_data:
        return pd.DataFrame(results_data).sort_values(by="Similarity Score", ascending=False)
    return pd.DataFrame()

# -------------------------------
# UI Layout (Updated)
# -------------------------------
col1, col2 = st.columns([1, 5])
with col1:
    # <<< --- THE FIX IS HERE: Using the local logo.png file --- >>>
    st.image("logo.png", width=150)
with col2:
    st.title("AI Duplication Model")
    st.write("Upload your file to detect duplicate work orders.")
st.markdown("---")

model = load_model()

uploaded_file = st.file_uploader("Choose your file", type=["xlsx"])
run = st.button("Start")

if uploaded_file is not None and run:
    with st.spinner("Analyzing file..."):
        dataframe = pd.read_excel(uploaded_file)
        results_df = find_similar_work_order_pairs(dataframe, model)
    st.success("Analysis Complete!")
    if not results_df.empty:
        st.subheader("Found Similarities")
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(label="Download results as CSV", data=csv, file_name="similar_work_orders.csv", mime="text/csv")
    else:
        st.warning("No similar work orders were found.")
