# 🦅 Advanced-RAPTOR: Recursive Summarization + Vector Search for Scalable RAG

**Advanced-RAPTOR** is a multi-stage, recursive document processing and retrieval system that integrates **RAPTOR-style summarization**, **clustering**, and **retrieval-augmented generation (RAG)** to enable accurate, context-rich answers from large PDF documents.

Built with **LangChain**, **FAISS**, **OpenAI**, and **FastAPI**, this system supports intelligent query answering over hierarchical document summaries, using a scalable and extensible architecture.

---

## 🚀 Key Features

- 🔁 **Recursive Summarization Tree** – Multi-level summarization of PDFs using LLMs and clustering
- 📚 **PDF Ingestion Pipeline** – Handles scanned and native PDFs using `PyMuPDF`, `OCRmyPDF`
- 🧠 **GMM Clustering** – Groups document chunks using Gaussian Mixture Models
- 🔍 **Contextual Retrieval** – Combines FAISS + LLM-based compression for precise answers
- 🧾 **Metadata-Aware Search** – Tracks hierarchical lineage across all RAPTOR levels
- ⚡ **FastAPI Interface** – Provides endpoints for document upload, vectorization, and QA
- 💡 **Interactive QA** – Command-line + API interface for real-time query responses

---

## 🧱 Tech Stack

- **LangChain** – Orchestrating summarization, chunking, and query pipelines
- **OpenAI GPT-4o-mini** – Summarization, compression, and final answer generation
- **FAISS** – High-performance vector similarity search
- **PyMuPDF / OCRmyPDF** – PDF text extraction with OCR support
- **Scikit-learn / GMM** – Embedding clustering for tree summarization
- **FastAPI** – Lightweight, extensible backend API
- **MongoDB (optional)** – For storing document metadata and user linkage

---

## 🗂 Project Structure

```
Advanced-RAPTOR/
│
├── askqa.py               # QA module using contextual compression retriever
├── raptor.py              # Recursive summarization and clustering (RAPTOR pipeline)
├── vectordb.py            # Vectorstore creation and FAISS persistence
├── main.py                # CLI execution entry point
├── data/                  # Sample PDFs
├── Vector_DB/             # Saved FAISS indexes
├── summaries/             # Generated summaries (level-wise)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/Advanced-RAPTOR.git
cd Advanced-RAPTOR
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Set OpenAI API key**

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_key_here
```

4. **Run the RAPTOR pipeline on a PDF**

```bash
python raptor.py
```

5. **Ask questions on the generated summaries**

```bash
python askqa.py
```

---

## 🔄 Workflow Overview

1. **Load PDF** ➝
2. **Extract pages & embed** ➝
3. **Cluster with GMM** ➝
4. **Summarize per cluster** ➝
5. **Repeat recursively** ➝
6. **Store all summaries in FAISS vector DB** ➝
7. **Answer questions using context-aware compression + LLM**

---

## 📊 Output

- 🔸 Hierarchical summaries in `summaries/`
- 🔹 FAISS vector store in `Vector_DB/`
- ✅ Answers from `askqa.py` CLI or API endpoints

---

## 🔮 Coming Soon

- Claude / Mistral / Open-Source LLM support  
- RAPTOR tree visualization (graph view)  
- Web frontend with chat-style interface  
- Multi-document & hybrid retrievers  
- Whisper-based audio summarization

---

## 📄 License

MIT License. For academic or personal use only. Commercial use requires explicit permission.

---

## 🙏 Acknowledgements

Inspired by:
- [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2310.02238)
- [LangChain](https://github.com/langchain-ai/langchain)
- [FAISS](https://github.com/facebookresearch/faiss)
- [OpenAI](https://openai.com/)
```
