import os
import streamlit as st
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment
load_dotenv()


def main():
    st.title("PDF Question Answering App")
    st.info("Ready to analyze your documents!")

    openai_api_key = st.text_input(" Enter OpenAI API Key :", type="password")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_pdf_path = tmp_file.name

        st.success("PDF file uploaded successfully!")

        # --- Load & split PDF ---
        try:
            loader = PyPDFLoader(temp_pdf_path)
            documents = loader.load()
        except Exception as e:
            st.error(f"Could not read PDF: {e}")
            st.stop()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        # --- Choose embeddings ---
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            st.info("Using OpenAI embeddings")
        else:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.info("Using HuggingFace embeddings (Free)")

        # --- Build FAISS index ---
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever()

        # --- Define prompt ---
        prompt = ChatPromptTemplate.from_template(
            """You are a helpful AI assistant. 
                Use the following context to answer the user's question accurately and concisely.
            Context:
                {context}
            Question:
                {question}    Answer:"""
        )

        # --- Select LLM ---
        if openai_api_key:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)
            st.info("Using OpenAI GPT-4o-mini")
        else:
            llm = ChatGroq(model="llama3-8b-8192")
            st.info("Using Groq Llama3-8b-8192")

        # --- Define the chain ---
        def retrieve_and_format(query):
            """Retrieve relevant documents and format them as plain text."""
            retrieved_docs = retriever.invoke(query)
            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
            return {"context": context_text, "question": query}

        # --- Combine everything ---
        chain = (
            RunnablePassthrough()
            | retrieve_and_format
            | prompt
            | llm
            | StrOutputParser()
        )

        # --- User query ---
        user_query = st.text_input("Ask a question about your PDF:")

        if user_query:
            with st.spinner("🤔 Thinking..."):
                try:
                    result = chain.invoke(user_query)
                    st.success("Response:")
                    st.write(result)
                except Exception as e:
                    st.error(f"Error during processing: {e}")


if __name__ == "__main__":
    main()
