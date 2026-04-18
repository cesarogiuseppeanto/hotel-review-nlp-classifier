import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from predict import predict_review

st.title("Analisi recensioni hotel")

text = st.text_area("Inserisci una recensione:")

if st.button("Analizza"):
    if text.strip() == "":
        st.warning("Inserisci del testo")
    else:
        # usiamo lo stesso testo sia come titolo che corpo
        dep, sent = predict_review(text, text)
        st.success(f"Reparto: {dep}")
        st.success(f"Sentiment: {sent}")
