import streamlit as st
import os, tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def main():
    history = []
    st.title("PDF Document Search")
    # Initialize Streamlit
    st.sidebar.title("Document Processing")
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)
    
    # Input for the question
    question = st.text_input("Enter your question")

    # Button to process uploaded files and question
    if st.button("Submit"):
        if uploaded_files:
            text = []
            for file in uploaded_files:
                file_extension = os.path.splitext(file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(file.read())
                    temp_file_path = temp_file.name

                loader = None
                if file_extension == ".pdf":
                    loader = PyPDFLoader(temp_file_path)

                if loader:
                    text.extend(loader.load())
                    os.remove(temp_file_path)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=50)
            text_chunks = text_splitter.split_documents(text)

            # Create embeddings
            embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5", 
                                            model_kwargs={'device': 'cpu'})

            # Create vector store
            vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
            
            docs = vector_store.similarity_search(question)
            
            # Append the question and results to the history
            history.append({"question": question, "results": docs})
            
            # Display results one by one
            for doc in docs:
                st.write(doc)
                
            # Display the history of questions and results
            if history:
                st.header("Previous Questions and Results")
                for entry in history:
                    st.subheader("Question:")
                    st.write(entry["question"])
                    st.subheader("Results:")
                    for doc in entry["results"]:
                        st.write(doc)


if __name__ == "__main__":
    main()
