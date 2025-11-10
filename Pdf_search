import streamlit as st
from langchain_classic.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_classic.embeddings import OpenAIEmbeddings
from langchain_classic.vectorstores import FAISS
from langchain_classic.llms import OpenAI
from langchain_classic import hub
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


# Streamlit app
def main():
 st.title("PDF Document Processing with Langchain")

 # File uploader
 uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

 if uploaded_file is not None:
 # Load the PDF document
 loader = PyPDFLoader(file_path=uploaded_file)
 documents = loader.load()

 # Split the documents into manageable chunks
 text_splitter = CharacterTextSplitter(
 chunk_size=1000, chunk_overlap=30, separator="\n"
 )
 docs = text_splitter.split_documents(documents=documents)

 # Create embeddings for the documents
 embeddings = OpenAIEmbeddings()
 vectorstore = FAISS.from_documents(docs, embeddings)
 vectorstore.save_local("faiss_index_react")

 # Load the vectorstore from local storage
 new_vectorstore = FAISS.load_local(
 "faiss_index_react", embeddings, allow_dangerous_deserialization=True
 )

 # Pull the retrieval QA chat prompt
 retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

 # Create the document chain
 combine_docs_chain = create_stuff_documents_chain(
 llm=OpenAI(), prompt=retrieval_qa_chat_prompt
 )

 # Create the retrieval chain
 retrieval_chain = create_retrieval_chain(
 retriever=new_vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
 )

 # User input for the query
 user_input = st.text_input("Enter your query:", "")

 if user_input:
 # Invoke the retrieval chain to get an answer
 res = retrieval_chain.invoke({"input": user_input})
 st.write("Response:", res["answer"])


if __name__ == "__main__":
 main()

