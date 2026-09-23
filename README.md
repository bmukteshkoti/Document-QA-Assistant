# 📚 Document-QA-Assistant

An AI-powered Document Question Answering Assistant that allows users to upload PDF or TXT documents and ask questions based on their content.

The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant document content and generate grounded answers using Google's Gemini API.

---

## 🚀 Features

- 📄 Upload PDF and TXT documents
- 🔍 Extract text from uploaded documents
- ✂️ Split documents into smaller overlapping chunks
- 🧠 Generate semantic embeddings using Sentence Transformers
- 📊 Store and search embeddings using FAISS
- 💬 Ask questions about uploaded documents
- 🤖 Generate answers using Google Gemini
- 📌 Answers are grounded in retrieved document content
- ⚠️ Handles empty questions and unreadable documents
- 🌐 Streamlit-based web interface
- 🔐 Secure API key management using environment variables and Streamlit Secrets

---

## 🏗️ Project Architecture

```text
User
  │
  ▼
Upload PDF / TXT
  │
  ▼
Document Text Extraction
  │
  ▼
Text Chunking
  │
  ▼
Sentence Transformer Embeddings
  │
  ▼
FAISS Vector Store
  │
  ▼
Question
  │
  ▼
Question Embedding
  │
  ▼
Similarity Search
  │
  ▼
Top Relevant Chunks
  │
  ▼
Gemini API
  │
  ▼
Grounded Answer
