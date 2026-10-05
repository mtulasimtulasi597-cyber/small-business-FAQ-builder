import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

# -----------------------------
# Page Settings
# -----------------------------
st.set_page_config(
    page_title="General Store FAQ Chatbot",
    page_icon="🛒",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("🛒 General Store FAQ Chatbot")

st.write(
    "Welcome! Ask questions about store timings, "
    "products, delivery, payments and returns."
)

# -----------------------------
# Load FAQ File
# -----------------------------
@st.cache_resource
def load_faq():

    with open(
        "data/general_store_faq.txt",
        "r",
        encoding="utf-8"
    ) as file:
        text = file.read()

    # Find Question + Answer pairs
    pattern = r"Q:\s*(.*?)\s*A:\s*(.*?)(?=\nQ:|\Z)"

    matches = re.findall(
        pattern,
        text,
        re.DOTALL
    )

    questions = []
    answers = []

    for question, answer in matches:
        questions.append(question.strip())
        answers.append(answer.strip())

    # Load embedding model
    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    # Convert questions into embeddings
    embeddings = model.encode(
        questions,
        normalize_embeddings=True
    )

    embeddings = np.array(
        embeddings
    ).astype("float32")

    # Create FAISS index
    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)

    return questions, answers, model, index


# Load FAQ
questions, answers, model, index = load_faq()


# -----------------------------
# User Question
# -----------------------------
user_question = st.text_input(
    "Ask your question:",
    placeholder="Example: Do you provide home delivery?"
)


# -----------------------------
# Ask Button
# -----------------------------
if st.button("Ask Question"):

    if user_question.strip() == "":
        st.warning("Please enter a question.")

    else:

        # Convert user question into embedding
        question_embedding = model.encode(
            [user_question],
            normalize_embeddings=True
        )

        question_embedding = np.array(
            question_embedding
        ).astype("float32")

        # Search FAISS
        scores, results = index.search(
            question_embedding,
            1
        )

        best_index = results[0][0]
        best_score = scores[0][0]

        # -----------------------------
        # Display Answer
        # -----------------------------
        if best_score >= 0.35:

            st.subheader("Answer")

            st.success(
                answers[best_index]
            )

            # Optional: show matched FAQ
            with st.expander("View matched FAQ"):
                st.write(
                    questions[best_index]
                )

        else:

            st.warning(
                "Sorry, I could not find an answer "
                "in our store FAQ."
            )