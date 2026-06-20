from pathlib import Path
import streamlit as st
import joblib

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "cine_sense_model.pkl"

# Load model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Page config
st.set_page_config(
    page_title="CineSense AI",
    page_icon="🎬",
    layout="wide"
)
with st.sidebar:
    st.header("Project Details")

    st.metric("Macro F1 Score", "0.3917")
    st.metric("Accuracy", "56.93%")

    st.write("---")

    st.write("**Model:** Balanced Linear SVM")
    st.write("**Features:** TF-IDF (1–2 grams)")
    st.write("**Training Samples:** 54,214")
    st.write("**Genres:** 27")
    
st.title("🎬 CineSense AI")
st.subheader("Explainable Movie Genre Classification using NLP")

st.markdown(
    """
Paste a movie plot summary below and let AI predict its genre.
"""
)

if "example" not in st.session_state:
    st.session_state.example = (
        "A young astronaut travels through a wormhole in search of a new home for humanity."
    )

st.markdown("### Try an example")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("👻 Horror"):
        st.session_state.example = (
            "A group of teenagers encounter a terrifying creature while camping in the woods."
        )

with col2:
    if st.button("🚀 Sci-Fi"):
        st.session_state.example = (
            "A young astronaut travels through a wormhole in search of a new home for humanity."
        )

with col3:
    if st.button("😂 Comedy"):
        st.session_state.example = (
            "Two best friends accidentally start a catering business that leads to hilarious situations."
        )

plot = st.text_area(
    "Movie Plot",
    value=st.session_state.example,
    height=200
)


if st.button("Predict Genre", use_container_width=True):

    if plot.strip():

        prediction = model.predict([plot])[0]

        st.success(f"Predicted Genre: **{prediction.upper()}**")

        st.info(
            "This prediction is generated using a TF-IDF + Balanced Linear SVM pipeline."
        )

    else:
        st.warning("Please enter a movie plot.")