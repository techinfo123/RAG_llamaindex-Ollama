import streamlit as st
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import fitz  # PyMuPDF

# Extract text from uploaded PDF file
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Cache the model loading and indexing
@st.cache_resource
def load_engine_from_text(text):
    # Convert raw text into LlamaIndex Document
    documents = [Document(text=text)]

    # Set embedding model
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    # Set local Ollama LLM
    Settings.llm = Ollama(
        model="mistral",
        base_url="http://localhost:11434",
        request_timeout=300.0
    )

    # Build index and return query engine
    index = VectorStoreIndex.from_documents(documents)
    return index.as_query_engine()

# Streamlit UI
st.set_page_config(page_title="PDF Q&A with Ollama + LlamaIndex")
st.title("📄 PDF Q&A using Ollama + LlamaIndex")

uploaded_pdf = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_pdf:
    with st.spinner("Reading and indexing PDF..."):
        text = extract_text_from_pdf(uploaded_pdf)
        query_engine = load_engine_from_text(text)

    question = st.text_input("Ask a question about your PDF:")

    if st.button("Submit"):
        if question.strip() == "":
            st.warning("Please enter a question.")
        else:
            with st.spinner("Querying PDF..."):
                response = query_engine.query(question)
                st.success("Answer:")
                st.write(response)
