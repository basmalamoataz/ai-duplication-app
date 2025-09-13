
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
    page_icon="logo_colour.png", # Updated to use the image file
    layout="wide"
)

# -------------------------------
# Inject custom CSS
# -------------------------------
CUSTOM_CSS = """
<style>
/* Global */
:root{
  --primary:#3B82F6;         /* blue-500 */
  --primary-600:#2563EB;     /* blue-600 */
  --bg:#0b1220;              /* dark navy background */
  --panel:#121a2b;           /* card background */
  --text:#E5E7EB;            /* gray-200 */
  --muted:#9CA3AF;           /* gray-400 */
  --accent:#10B981;          /* emerald-500 */
  --danger:#EF4444;          /* red-500 */
  --radius:14px;
}

html, body, [class^="stApp"] {
  background: radial-gradient(1200px 800px at 10% -20%, rgba(59,130,246,0.12), transparent 60%) ,
              radial-gradient(1000px 700px at 110% 10%, rgba(16,185,129,0.10), transparent 50%) ,
              var(--bg);
  color: var(--text);
  font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
}

/* Title */
h1, h2, h3 {
  letter-spacing: 0.2px;
}
h1 {
  font-weight: 800;
  font-size: clamp(28px, 3.2vw, 44px);
  margin-bottom: 0.4rem;
}
p {
  color: var(--muted);
}

/* Containers (cards) */
.block-container {
  padding-top: 1.5rem;
  max-width: 1200px;
}
.stApp > header {background: transparent;}
/* Make widgets look like cards */
div[data-testid="stFileUploader"] > section,
div[data-testid="stDataFrame"] {
  background: var(--panel);
  border: 1px solid rgba(255,255,255,0.06);
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
  width: 900px;
  border-radius: var(--radius);
}

/* Buttons */
div[data-testid="stFileUploader"] button {
  background: var(--panel) !important;   /* dark */
  color: var(--text) !important;
  border: 1px solid rgba(255,255,255,0.15) !important;
  border-radius: 10px !important;
  padding: 0.6rem 1rem !important;
  font-weight: 600 !important;
  box-shadow: 0 4px 12px rgba(0,0,0,0.35) !important;
  transition: background .2s ease, transform .1s ease;
}

div[data-testid="stFileUploader"] button:hover {
  background: rgba(255,255,255,0.1) !important;
  transform: translateY(-1px);
}

div[data-testid="stFileUploader"] button:active {
  background: rgba(255,255,255,0.2) !important;
  transform: translateY(1px);
}

/* Start button - blue accent */
.stButton > button {
  background: linear-gradient(135deg, var(--primary), var(--primary-600)) !important;
  color: white !important;
  border: 0 !important;
  border-radius: 10px !important;
  padding: 0.7rem 1.1rem !important;
  font-weight: 600 !important;
  transition: transform .05s ease, box-shadow .2s ease, filter .2s ease;
  box-shadow: 0 10px 20px rgba(37,99,235,0.25) !important;
}

.stButton > button:hover {
  filter: brightness(1.08) !important;
}

.stButton > button:active {
  transform: translateY(1px) !important;
}


/* File uploader tweaks */
div[data-testid="stFileUploader"] {
  padding: 1rem;

}
div[data-testid="stFileUploader"] label {
  font-weight: 600;
  color: var(--text);
}
div[data-testid="stFileUploader"] button {
  background: var(--panel) !important;
  color: var(--text) !important;
  border: 1px solid rgba(255,255,255,0.15) !important;
  border-radius: 10px !important;
  padding: 0.6rem 1rem !important;
  font-weight: 600 !important;
  box-shadow: 0 4px 12px rgba(0,0,0,0.35) !important;
  transition: background .2s ease, transform .1s ease;
}

div[data-testid="stFileUploader"] button:hover {
  background: rgba(255,255,255,0.1) !important;
  transform: translateY(-1px);
}

div[data-testid="stFileUploader"] button:active {
  background: rgba(255,255,255,0.2) !important;
  transform: translateY(1px);
}

/* Status messages */
.stAlert {
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,0.08);
}

/* Dataframe: dark, compact */
[data-testid="stTable"], [data-testid="stDataFrame"] {
  overflow: hidden;
}
[data-testid="stDataFrame"] div[role="table"] {
  color: var(--text);
}
[data-testid="stDataFrame"] .css-1kz0wi6,  /* header cells (Streamlit internal class may change) */
[data-testid="stDataFrame"] .index_name {
  background: rgba(255,255,255,0.04) !important;
}
[data-testid="stDataFrame"] th, [data-testid="stDataFrame"] td {
  border-color: rgba(255,255,255,0.08) !important;
}
[data-testid="StyledDataFrame"] {
  font-size: 0.92rem;
}

/* Download button */
[data-testid="baseButton-secondary"] {
  background: transparent !important;
  color: var(--text) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  border-radius: 10px !important;
}

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
# Analysis
# -------------------------------
def find_similar_work_order_pairs(df, model):
    desc_column = "Description"
    wo_number_column = "Work Order Number"
    status_column = "Status"
    location_column = "Location"
    asset_column = "Asset"
    classification_column = "Classification"
    pm_number_column = "PM Number"

    statuses_to_exclude = ["COMP", "CAN"]
    similarity_threshold = 0.75

    if status_column in df.columns:
        df[status_column] = df[status_column].astype(str)

    filtered_df = df
    if status_column in df.columns:
        filtered_df = df[~df[status_column].isin(statuses_to_exclude)]
    else:
        st.warning(f"Column '{status_column}' not found. Skipping status filter.")

    if pm_number_column in filtered_df.columns:
        filtered_df[pm_number_column] = filtered_df[pm_number_column].astype(str)
        filtered_df = filtered_df[
            filtered_df[pm_number_column].isnull() | (filtered_df[pm_number_column] == 'nan')
        ].copy()
    else:
        st.warning(f"Column '{pm_number_column}' not found. Skipping PM filter.")

    results_data = []
    if not {location_column, asset_column}.issubset(filtered_df.columns):
        st.error("Missing required columns for grouping: Location and/or Asset.")
        return pd.DataFrame()

    grouped = filtered_df.groupby([location_column, asset_column])

    for (location, asset), group_df in grouped:
        if len(group_df) < 2:
            continue

        sentences_df = group_df.dropna(subset=[desc_column])
        if len(sentences_df) < 2:
            continue

        sentences = sentences_df[desc_column].astype(str).tolist()
        work_orders = sentences_df[wo_number_column].tolist()

        embeddings = model.encode(sentences, show_progress_bar=False, convert_to_tensor=True)
        clusters = util.community_detection(
            embeddings, threshold=similarity_threshold, min_community_size=2
        )

        if clusters:
            for cluster in clusters:
                for i_pos in range(len(cluster)):
                    for j_pos in range(i_pos + 1, len(cluster)):
                        idx1 = cluster[i_pos]
                        idx2 = cluster[j_pos]
                        score = util.cos_sim(embeddings[idx1], embeddings[idx2])[0, 0].item()
                        results_data.append(
                            {
                                "Similarity Score": f"{score:.4f}",
                                "Work Order 1": work_orders[idx1],
                                "Description 1": sentences[idx1],
                                "Work Order 2": work_orders[idx2],
                                "Description 2": sentences[idx2],
                                "Location": location,
                                "Asset": asset,
                            }
                        )

    if results_data:
        return pd.DataFrame(results_data).sort_values(by="Similarity Score", ascending=False)
    return pd.DataFrame()

# -------------------------------
# UI
# -------------------------------
# --- Header Section ---


st.title("AI Duplication Model")
st.write("Upload your file to detect duplicate work orders.")
st.markdown("---")


model = load_model()

left, right = st.columns([1, 2])
with left:
    uploaded_file = st.file_uploader("Choose your file", type=["xlsx"])
    run = st.button("Start")

if uploaded_file is not None and run:
    with st.spinner("Analyzing file..."):
        dataframe = pd.read_excel(uploaded_file)
        results_df = find_similar_work_order_pairs(dataframe, model)

    st.success("Analysis Complete!")

    if not results_df.empty:
        st.subheader("Found Similarities")
        st.dataframe(results_df, use_container_width=True)
        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download results as CSV",
            data=csv,
            file_name="similar_work_orders.csv",
            mime="text/csv",
        )
    else:
        st.warning("No similar work orders were found with the current settings.")
