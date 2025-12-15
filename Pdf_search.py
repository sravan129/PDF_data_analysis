# pdf_app.py
import streamlit as st
import tempfile
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain.chains.combine_documents import create_combine_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


# Streamlit App
def main():
    st.title("📘 PDF Question Answering App (LangChain v1.x)")

    uploaded_file = st.file_uploader("📂 Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        # Save to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_pdf_path = tmp_file.name

        st.info("✅ PDF file uploaded successfully!")

        # Load PDF
        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()

        # Split text into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        # Create embeddings and FAISS index
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)

        # Create retriever
        retriever = vectorstore.as_retriever()

        # Pull Retrieval QA prompt from LangChain Hub
        retrieval_qa_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

        # Initialize LLM
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Combine document chain (new API)
        combine_chain = create_combine_documents_chain(llm=llm, prompt=retrieval_qa_prompt)

        # Create retrieval chain (new API)
        retrieval_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=combine_chain)

        # User query input
        user_query = st.text_input("💬 Ask a question about the PDF:")

        if user_query:
            with st.spinner("Thinking..."):
                result = retrieval_chain.invoke({"input": user_query})
            st.success("✅ Response:")
            st.write(result["answer"])


if __name__ == "__main__":
    main()
