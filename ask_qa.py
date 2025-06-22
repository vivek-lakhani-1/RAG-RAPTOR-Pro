import os
import warnings
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import ChatPromptTemplate
from langchain.chains.llm import LLMChain
from langchain_openai import ChatOpenAI

# 🔇 Disable warnings and logging from LangChain/OpenAI
warnings.filterwarnings("ignore")
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_TRACING"] = "false"

def load_retriever(vector_path: str, llm: ChatOpenAI):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        vector_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    base_retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_template(
        "Given the following context and question, extract only the relevant information for answering the question:\n\n"
        "Context: {context}\n"
        "Question: {question}\n\n"
        "Relevant Information:"
    )
    extractor = LLMChainExtractor.from_llm(llm, prompt=prompt)
    return ContextualCompressionRetriever(
        base_compressor=extractor,
        base_retriever=base_retriever
    )

def answer_query(retriever, query: str, llm: ChatOpenAI):
    relevant_docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = ChatPromptTemplate.from_template(
        "Given the following context, please answer the question:\n\n"
        "Context: {context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    answer = chain.run(context=context, question=query)

    return {
        "query": query,
        "context_used": context,
        "answer": answer,
        "model_used": llm.model_name,
    }

if __name__ == "__main__":
    vector_path = "Vector_DB/JP-Proxy"
    llm = ChatOpenAI(model_name="gpt-4o-mini")

    retriever = load_retriever(vector_path, llm)
    print("\n💬 Enter your queries (type 'q' to quit):\n")

    while True:
        query = input("❓ Your question: ").strip()
        if query.lower() == 'q':
            print("👋 Goodbye!")
            break

        result = answer_query(retriever, query, llm)

        print(f"\n🔍 Query: {result['query']}")
        print(f"\n📚 Context Used (truncated):\n{result['context_used'][:1500]}...\n")
        print(f"✅ Answer:\n{result['answer']}")
        print(f"\n🤖 Model Used: {result['model_used']}")
        print("-" * 80 + "\n")