import streamlit as st

from document_loader import extract_text_from_pdf, extract_text_from_txt
from chunking import split_text_into_chunks
from embeddings import create_embeddings
from vector_store import create_vector_store, search_similar_chunks
from qa import generate_answer


st.title("📚 Document Q&A Assistant")

uploaded_files = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if uploaded_files:

    all_chunks = []

    for file in uploaded_files:

        if file.name.lower().endswith(".pdf"):
            text = extract_text_from_pdf(file)
        else:
            text = extract_text_from_txt(file)

        if text.strip():

            chunks = split_text_into_chunks(text)

            all_chunks.extend(chunks)

            st.subheader(file.name)

            st.write("Extracted characters:", len(text))
            st.success(f"Document processed into {len(chunks)} chunks.")

        else:
            st.warning(f"No text could be extracted from {file.name}")

    if all_chunks:

        st.write("### Creating Embeddings...")

        embeddings = create_embeddings(all_chunks)

        st.success("Embeddings created successfully.")

        st.write("### Creating Vector Store...")

        index = create_vector_store(embeddings)

        st.success("FAISS vector store created successfully.")

        st.write("### Ask a Question")

        question = st.text_input(
            "Enter your question about the uploaded documents"
        )

        if question.strip():

            question_embedding = create_embeddings([question])[0]

            results = search_similar_chunks(
                index,
                question_embedding,
                all_chunks,
                top_k=3
            )

            answer = generate_answer(
                question,
                results
            )

            st.write("### Answer")

            st.write(answer)

            st.write("### Supporting Context")

            for i, result in enumerate(results):

                with st.expander(f"Source {i + 1}"):
                    st.write(result)

        elif question == "":
            st.info("Enter a question to get an answer.")