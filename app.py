import streamlit as st
import base64
from html import escape

from document_loader import extract_text_from_pdf, extract_text_from_txt
from chunking import split_text_into_chunks
from embeddings import create_embeddings
from vector_store import create_vector_store, search_similar_chunks
from qa import generate_answer


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# BACKGROUND
# ============================================================

background_path = "assets/background.png"

with open(background_path, "rb") as f:
    background_data = base64.b64encode(f.read()).decode()


# ============================================================
# CSS
# ============================================================

st.markdown(
    f"""
<style>

.stApp {{
    background-image:
        linear-gradient(
            rgba(245, 248, 255, 0.80),
            rgba(245, 248, 255, 0.80)
        ),
        url("data:image/png;base64,{background_data}");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}


/* ============================================================
   HIDE STREAMLIT DEFAULT HEADER
   ============================================================ */

#MainMenu {{
    display: none !important;
}}

footer {{
    display: none !important;
}}

[data-testid="stHeader"] {{
    display: none !important;
}}

[data-testid="stToolbar"] {{
    display: none !important;
}}

[data-testid="stDecoration"] {{
    display: none !important;
}}

[data-testid="stStatusWidget"] {{
    display: none !important;
}}


/* ============================================================
   PAGE WIDTH
   ============================================================ */

.block-container {{
    max-width: 1100px;
    padding-top: 20px !important;
    padding-bottom: 40px !important;
}}


/* ============================================================
   NAVBAR
   ============================================================ */

.navbar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 45px;
}}

.brand {{
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 25px;
    font-weight: 800;
    color: #172033;
}}

.brand-icon {{
    width: 42px;
    height: 42px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 12px;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    font-size: 21px;
}}

.nav-tag {{
    background: rgba(238, 242, 255, 0.95);
    color: #4f46e5;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
}}


/* ============================================================
   HERO
   ============================================================ */

.hero-badge {{
    text-align: center;
    color: #6366f1;
    font-size: 16px;
    font-weight: 700;
    margin: 0 0 10px 0;
}}

.hero-title {{
    text-align: center;
    color: #172033;
    font-size: 48px;
    line-height: 1.15;
    font-weight: 800;
    letter-spacing: -1px;
    margin: 0 0 42px 0;
}}


/* ============================================================
   SECTION TITLES
   ============================================================ */

.section-title {{
    color: #172033;
    font-size: 24px;
    font-weight: 800;
    margin-bottom: 12px;
}}


/* ============================================================
   FILE UPLOADER
   ============================================================ */

[data-testid="stFileUploader"] {{
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    box-shadow: none !important;
}}

[data-testid="stFileUploaderDropzone"] {{
    background: rgba(255, 255, 255, 0.88) !important;
    border: 2px dashed #a5b4fc !important;
    border-radius: 16px !important;
}}


/* ============================================================
   SUCCESS MESSAGE
   ============================================================ */

.success-box {{
    background: rgba(236, 253, 245, 0.96);
    border: 1px solid #a7f3d0;
    color: #047857;
    padding: 16px 20px;
    border-radius: 14px;
    margin: 25px 0;
    font-weight: 700;
}}


/* ============================================================
   QUESTION
   ============================================================ */

.question-card {{
    background: rgba(255, 255, 255, 0.94);
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 8px 25px rgba(30, 41, 59, 0.07);
}}

.stTextInput input {{
    height: 50px;
    border-radius: 10px;
    font-size: 16px;
}}

.stFormSubmitButton button {{
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 25px;
    font-weight: 700;
}}


/* ============================================================
   ANSWER
   ============================================================ */

.answer-card {{
    background: rgba(255, 255, 255, 0.95);
    border-left: 5px solid #6366f1;
    border-radius: 15px;
    padding: 22px;
    color: #334155;
    line-height: 1.7;
    box-shadow: 0 8px 25px rgba(30, 41, 59, 0.07);
}}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {{
    text-align: center;
    color: #64748b;
    font-size: 13px;
    margin-top: 50px;
    padding-top: 20px;
    border-top: 1px solid rgba(203, 213, 225, 0.6);
}}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# NAVBAR
# ============================================================

st.markdown(
    '<div class="navbar"><div class="brand"><div class="brand-icon">📚</div><div>DocuMind AI</div></div><div class="nav-tag">AI • RAG ASSISTANT</div></div>',
    unsafe_allow_html=True
)


# ============================================================
# HERO TEXT
# ============================================================

st.markdown(
    '<div class="hero-badge">✨ Intelligent Document Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-title">Turn Your Documents into QnA</div>',
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD DOCUMENTS
# ============================================================

st.markdown(
    '<div class="section-title">📄 Upload Documents</div>',
    unsafe_allow_html=True
)

uploaded_files = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

if uploaded_files:

    all_chunks = []
    total_characters = 0
    valid_documents = 0

    for file in uploaded_files:

        if file.name.lower().endswith(".pdf"):
            text = extract_text_from_pdf(file)

        elif file.name.lower().endswith(".txt"):
            text = extract_text_from_txt(file)

        else:
            continue

        if text and text.strip():

            valid_documents += 1
            total_characters += len(text)

            chunks = split_text_into_chunks(text)

            all_chunks.extend(chunks)

        else:

            st.warning(
                f"No readable text found in {file.name}"
            )


    # ========================================================
    # EMBEDDINGS + VECTOR STORE
    # ========================================================

    if all_chunks:

        with st.spinner("Preparing your documents..."):

            embeddings = create_embeddings(
                all_chunks
            )

            index = create_vector_store(
                embeddings
            )


        st.markdown(
            '<div class="success-box">✓ Documents processed successfully. Your AI assistant is ready.</div>',
            unsafe_allow_html=True
        )

        # ====================================================
        # ASK QUESTION
        # ====================================================

        st.markdown(
            '<div class="section-title">💬 Ask Your Documents</div>',
            unsafe_allow_html=True
        )

        with st.form("question_form"):

            question = st.text_input(
                "What would you like to know?",
                placeholder="Example: What is cloud computing?"
            )

            submitted = st.form_submit_button(
                "🔍 Ask Question"
            )


        # ====================================================
        # QUESTION PROCESSING
        # ====================================================

        if submitted:

            if not question.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                with st.spinner(
                    "Searching your documents..."
                ):

                    question_embedding = create_embeddings(
                        [question]
                    )[0]

                    results = search_similar_chunks(
                        index,
                        question_embedding,
                        all_chunks,
                        top_k=3
                    )


                with st.spinner(
                    "Generating a grounded answer..."
                ):

                    answer = generate_answer(
                        question,
                        results
                    )


                from html import escape

                safe_answer = escape(answer).replace("\n", "<br>")

                st.markdown(
                    f'<div class="answer-card">{safe_answer}</div>',
                    unsafe_allow_html=True
                )

# ============================================================
# NO DOCUMENT MESSAGE
# ============================================================

else:

    st.info(
        "Upload at least one PDF or TXT document to begin."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">DocuMind AI • Retrieval-Augmented Question Answering</div>',
    unsafe_allow_html=True
)