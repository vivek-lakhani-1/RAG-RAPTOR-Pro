import os
import numpy as np
import pandas as pd
from typing import List, Dict
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from sklearn.mixture import GaussianMixture
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import logging
import warnings
warnings.filterwarnings("ignore")

# Disable LangChain/OpenAI tracing
os.environ["LANGCHAIN_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Load OpenAI API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found. Set it in your .env file.")
os.environ["OPENAI_API_KEY"] = api_key

def extract_text(item):
    from langchain.schema import AIMessage
    return item.content if isinstance(item, AIMessage) else item

def embed_texts(texts: List[str]) -> List[List[float]]:
    print(f"🔗 Embedding {len(texts)} texts...")
    embeddings = OpenAIEmbeddings()
    return embeddings.embed_documents([extract_text(t) for t in texts])

def perform_clustering(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    print(f"📊 Clustering into {n_clusters} clusters...")
    if len(embeddings) < 2:
        print("⚠️ Not enough samples to cluster — assigning to one cluster.")
        return np.array([0] * len(embeddings))
    gm = GaussianMixture(n_components=n_clusters, random_state=42)
    return gm.fit_predict(embeddings)

def summarize_texts(texts: List[str], llm: ChatOpenAI) -> str:
    print(f"📝 Summarizing {len(texts)} texts...")
    prompt = ChatPromptTemplate.from_template("Summarize the following text concisely:\n\n{text}")
    chain = prompt | llm
    return chain.invoke({"text": texts})

def build_raptor_tree(texts: List[str], llm: ChatOpenAI, max_levels: int = 3) -> Dict[int, pd.DataFrame]:
    print("🌲 Starting RAPTOR tree generation...")
    results = {}
    current_texts = texts
    current_metadata = [{"level": 0, "origin": "original", "parent_id": None} for _ in texts]

    for level in range(1, max_levels + 1):
        print(f"\n🔁 Processing level {level}...")
        if len(current_texts) <= 1:
            print("⚠️ Only one text remaining — stopping recursion.")
            results[level] = pd.DataFrame({
                'text': current_texts,
                'embedding': embed_texts(current_texts),
                'cluster': [0],
                'metadata': current_metadata
            })
            break

        embeddings = embed_texts(current_texts)
        n_clusters = max(1, min(10, len(current_texts) // 2))
        cluster_labels = perform_clustering(np.array(embeddings), n_clusters)

        df = pd.DataFrame({
            'text': current_texts,
            'embedding': embeddings,
            'cluster': cluster_labels,
            'metadata': current_metadata
        })
        results[level - 1] = df

        summaries, new_metadata = [], []
        for cluster in df['cluster'].unique():
            print(f"🔹 Summarizing cluster {cluster} at level {level - 1}")
            cluster_docs = df[df['cluster'] == cluster]
            summary = summarize_texts(cluster_docs['text'].tolist(), llm)
            summaries.append(summary)
            new_metadata.append({
                "level": level,
                "origin": f"summary_of_cluster_{cluster}_level_{level - 1}",
                "child_ids": [meta.get('id') for meta in cluster_docs['metadata'].tolist()],
                "id": f"summary_{level}_{cluster}"
            })

        current_texts = summaries
        current_metadata = new_metadata

        if len(current_texts) <= 1:
            print("✅ Final summary node created.")
            results[level] = pd.DataFrame({
                'text': current_texts,
                'embedding': embed_texts(current_texts),
                'cluster': [0],
                'metadata': current_metadata
            })
            break

    print("🌳 RAPTOR tree construction complete.")
    return results

def save_vectorstore(tree_results: Dict[int, pd.DataFrame], embeddings, save_path="Vector_DB"):
    print(f"💾 Saving FAISS vector store to: {save_path}")
    all_texts, all_embeddings, all_metadatas = [], [], []

    for df in tree_results.values():
        all_texts.extend(df['text'].tolist())
        all_embeddings.extend([e.tolist() if isinstance(e, np.ndarray) else e for e in df['embedding'].tolist()])
        all_metadatas.extend(df['metadata'].tolist())

    documents = [Document(page_content=str(t), metadata=m) for t, m in zip(all_texts, all_metadatas)]
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(save_path)
    print("✅ Vector store saved successfully.\n")

# ✅ Save summaries per level to .txt files
def save_summaries_to_txt(tree_results: Dict[int, pd.DataFrame], output_dir="summaries"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"🗂 Saving summaries to '{output_dir}'...")
    for level, df in tree_results.items():
        level_path = os.path.join(output_dir, f"level_{level}.txt")
        with open(level_path, "w", encoding="utf-8") as f:
            for i, text in enumerate(df['text'].tolist()):
                f.write(f"--- Summary {i + 1} ---\n{text}\n\n")
        print(f"📝 Level {level} summaries saved to {level_path}")

def print_clusters_per_level(tree_results: Dict[int, pd.DataFrame]):
    print("\n📌 Cluster Details by Level:")
    for level, df in tree_results.items():
        print(f"\n🔸 Level {level}:")
        for cluster_id in sorted(df['cluster'].unique()):
            print(f"\n  ▪️ Cluster {cluster_id}:")
            cluster_texts = df[df['cluster'] == cluster_id]['text'].tolist()
            for i, t in enumerate(cluster_texts):
                text = extract_text(t)
                clean_text = text[:200].replace('\n', ' ') if isinstance(text, str) else str(text)[:200]
                print(f"    {i + 1}. {clean_text}...")
# 🔽 Main execution
if __name__ == "__main__":
    pdf_path = "test/JP-Proxy.pdf"
    max_levels = 3
    filename = pdf_path.split('/')[-1].replace('.pdf', '')
    save_path = os.path.join("Vector_DB", filename)

    print(f"📄 Loading PDF from: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    texts = [doc.page_content for doc in documents]
    print(f"📚 Loaded {len(texts)} pages from PDF.")

    print("🤖 Initializing OpenAI model...")
    llm = ChatOpenAI(model_name="gpt-4o-mini")

    tree_results = build_raptor_tree(texts, llm, max_levels=max_levels)
    save_vectorstore(tree_results, OpenAIEmbeddings(), save_path=save_path)
    save_summaries_to_txt(tree_results)  # Save summaries
    print_clusters_per_level(tree_results)  # Print clusters
