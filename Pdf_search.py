import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# Streamlit App
def main():
    st.title("📘 PDF Document Processing with LangChain")

    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_pdf_path = tmp_file.name

        st.info("✅ PDF file uploaded successfully!")

        # --- Load the PDF ---
        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()

        # --- Split text into chunks ---
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
        docs = text_splitter.split_documents(documents)

        # --- Create embeddings and FAISS index ---
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local("faiss_index_react")

        # --- Load the saved vectorstore ---
        new_vectorstore = FAISS.load_local(
            "faiss_index_react", embeddings, allow_dangerous_deserialization=True
        )

        # --- Load Retrieval QA Prompt ---
        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

        # --- Create document and retrieval chains ---
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=retrieval_qa_chat_prompt)
        retrieval_chain = create_retrieval_chain(
            retriever=new_vectorstore.as_retriever(),
            combine_docs_chain=combine_docs_chain
        )

        # --- Query Input ---
        user_input = st.text_input("💬 Enter your query:", "")

        if user_input:
            with st.spinner("Thinking..."):
                res = retrieval_chain.invoke({"input": user_input})
            st.success("✅ Response:")
            st.write(res["answer"])


if __name__ == "__main__":
    main()
