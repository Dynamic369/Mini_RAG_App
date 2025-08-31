import streamlit as st
import os
import time
import uuid

from dotenv import load_dotenv
from pinecone import Pinecone

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Loading the environment
load_dotenv()
os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")

#Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "database"

# LLM and Prompt
llm = ChatNVIDIA(model='meta/llama-3.1-70b-instruct')

prompt = ChatPromptTemplate.from_template(
    """ 
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question.
        <context>
        {context}
        <context>
        Question: {input}
    """
)


#Create Namespace (per session/user)
if "namespace" not in st.session_state:
    st.session_state.namespace = f"session-{uuid.uuid4().hex[:8]}"


#Vector Embedding Function
def vector_embedding(pdf_files):
    if pdf_files:
        st.session_state.embeddings = NVIDIAEmbeddings()
        all_docs = []

        for uploaded_file in pdf_files:
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            all_docs.extend(docs)
            os.remove(temp_path)  # cleanup temp file

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=120
        )
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(all_docs)

        # Push embeddings into Pinecone with unique namespace
        st.session_state.vectors = PineconeVectorStore.from_documents(
            documents=st.session_state.final_documents,
            embedding=st.session_state.embeddings,
            index_name=index_name,
            namespace=st.session_state.namespace
        )
        return True
    return False


#Retriever + Reranker
def get_retriever_with_reranker():
    # Connect the pinecone and turn it into a retriever.
    base_retriever = PineconeVectorStore(
        index_name=index_name,
        embedding=NVIDIAEmbeddings(),
        namespace=st.session_state.namespace
    ).as_retriever(search_kwargs={"k": 5}) # shows top 5 similar search.

    reranker = LLMChainExtractor.from_llm(llm)
    # Wraping the base_retriever and reranker into single retriever.
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever
    )
    return compression_retriever


#Streamlit UI
st.title("📄 Mini RAG App")

# Button to clear session state
if st.button("Clear Session State"):
    for key in ["embeddings", "text_splitter", "final_documents", "vectors", "namespace"]:
        if key in st.session_state:
            del st.session_state[key]
    st.success("Session state cleared. Refresh to start fresh.")

# File uploader(can accept multiple files)
pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if st.button('Documents Embedding'):
    if pdf_files:
        if vector_embedding(pdf_files):
            st.success(f"✅ Vector DB ready in Pinecone (namespace: {st.session_state.namespace})")
    else:
        st.warning("⚠️ Please upload at least one PDF file first.")


#TEXT PASTE SECTION 
st.subheader("✍️ Or Paste Text")

user_text = st.text_area("Paste text here (instead of uploading a PDF):", height=200)

if st.button("Embed Pasted Text"):
    if user_text.strip():
        st.session_state.embeddings = NVIDIAEmbeddings()

        # Treat pasted text as one document
        from langchain.schema import Document
        docs = [Document(page_content=user_text, metadata={"source": "user_pasted"})]

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=120
        )
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(docs)

        st.session_state.vectors = PineconeVectorStore.from_documents(
            documents=st.session_state.final_documents,
            embedding=st.session_state.embeddings,
            index_name=index_name,
            namespace=st.session_state.namespace
        )
        st.success(f"✅ Pasted text embedded into Pinecone (namespace: {st.session_state.namespace})")
    else:
        st.warning("⚠️ Please paste some text first.")


# Question input
question = st.text_input("Ask a question based on the uploaded documents:")

 
# Answer Retrieval
if question:
    if 'vectors' in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt=prompt)
        retriever = get_retriever_with_reranker()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({"input": question})
        elapsed = time.process_time() - start

        st.write("### Answer:")
        st.write(response['answer'])
        st.caption(f"⏱️ Response time: {elapsed:.2f}s")

        with st.expander("🔍 Document similarity search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write("-------------------------------------------")
    else:
        st.warning("⚠️ Please embed a PDF document first or provide the text.")




