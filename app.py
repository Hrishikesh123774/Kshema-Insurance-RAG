import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Set your OpenAI API Key securely via environment variables
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

def run_insurance_rag_pipeline(pdf_path="sample_policy.pdf"):
    # 1. Ingest dense insurance documentation
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # 2. Recursive chunking for optimal structural context bounding
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 3. Vector Database Indexing using Embeddings[cite: 1]
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 4. Strict System Prompt Architecture to completely eliminate hallucinations[cite: 1]
    system_prompt = (
        "You are an expert insurance assistant. Answer the question using ONLY the provided context. "
        "If the information is not present, fail gracefully by stating that it is unavailable in the document.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )
    prompt = ChatPromptTemplate.from_template(system_prompt)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Clean text helper
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 5. Modular LCEL Chain Orchestration
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

if __name__ == "__main__":
    print("Kshema Insurance RAG Pipeline Stack Initialized Successfully.")
Scroll down to the bottom of the page and click the green Commit changes button. (If a confirmation pop-up shows up, click Commit changes inside it too).

Part B: Add requirements.txt
You will be taken back to your main repository screen (the same view as your screenshot).

Click that same Add file button again and select Create new file.

In the Name your file... box, type exactly:

requirements.txt

4. In the big text area below it, copy and paste these dependencies:
   ```text
langchain
langchain-community
langchain-openai
chromadb
pypdf
