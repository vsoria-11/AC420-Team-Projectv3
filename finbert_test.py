# [AI-GENERATED]
"""Tiny deployment test: can FinBERT load and run on Streamlit Community Cloud?
Deploy this file as its OWN app (main file path: finbert_test.py) BEFORE trusting the main app with FinBERT."""
import resource
import time

import streamlit as st

st.title("FinBERT deployment test")
st.caption("Streamlit Community Cloud allows roughly 2.7 GB of memory (the limit can change). "
           "If peak memory below is well under that, FinBERT should be safe to use in the main app.")

HEADLINES = [
    "Company beats earnings expectations and raises full-year guidance",
    "Shares plunge after regulators open fraud probe",
    "Company announces date for annual shareholder meeting",
]


def peak_memory_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024     # Linux reports kilobytes


st.write(f"Peak memory before loading: {peak_memory_mb():,.0f} MB")

if st.button("Load FinBERT and score 3 headlines", type="primary"):
    try:
        start = time.time()
        with st.spinner("Loading FinBERT (the first time downloads about 440 MB)..."):
            from transformers import pipeline
            classifier = pipeline("text-classification", model="ProsusAI/finbert", top_k=None)
        st.success(f"Model loaded in {time.time() - start:.0f} seconds")

        start = time.time()
        outputs = classifier(HEADLINES, truncation=True)
        for title, labels in zip(HEADLINES, outputs):
            probs = {item["label"].lower(): item["score"] for item in labels}
            st.write(f"**{probs['positive'] - probs['negative']:+.2f}**  {title}")
        st.write(f"Scored in {time.time() - start:.1f} seconds")
        st.write(f"Peak memory after loading: {peak_memory_mb():,.0f} MB")
    except Exception as error:
        st.error("FinBERT could not run here. The main app will fall back to VADER. Error: " + str(error)[:300])
