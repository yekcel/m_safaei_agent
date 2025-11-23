import streamlit as st
import os
import logging
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY if GEMINI_API_KEY else ""
#os.environ["GEMINI_API_KEY"] = API_KEY_VALUE


@st.cache_resource
def setup_rag_engine():        
    global GEMINI_API_KEY
    
    st.info("Loading resume data and indexing...")
    logger.info("Starting RAG setup process...")
    try:
        if not GEMINI_API_KEY:
            st.error("Error: GEMINI_API_KEY not found in Streamlit Secrets. Please configure it.") 
            return None
       
        logger.info("STEP 1: Attempting to load documents from 'data' folder.")
        documents = SimpleDirectoryReader("data").load_data()
        logger.info(f"Loaded {len(documents)} documents successfully.")
        # ----------------------------------------------------------------------------------  
        logger.info("STEP 1.5: Configuring Text Splitter for API stability.")
        text_splitter = SentenceSplitter(chunk_size=2048, chunk_overlap=20)
        nodes = text_splitter.get_nodes_from_documents(documents)
        logger.info(f"Documents split into {len(nodes)} chunks.")
        try:
            logger.info("STEP 2: Configuring GoogleGenAI models.")
            llm = GoogleGenAI(model="gemini-2.5-flash", api_key=GEMINI_API_KEY)
            embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5"         
            )
            logger.info("LLM configured (Gemini) and Embedding model configured (Local).")            
        except Exception as e:
            raise Exception(f"Error during model configuration: {e}")
        # ----------------------------------------------------------------------------------
        try:
            logger.info("STEP 3: Starting VectorStoreIndex creation (Chunking and Embedding).")          
            index = VectorStoreIndex(
                nodes=nodes, 
                llm=llm,              
                embed_model=embed_model 
            )
            logger.info("VectorStoreIndex created successfully.")
            
        except Exception as e:
            logger.error(f"FATAL ERROR IN STEP 3 (Indexing): {type(e).__name__} - {e}")
            raise Exception(f"Error during Indexing/Embedding: {e}")
                  
        try:
            logger.info("STEP 4: Creating Query Engine.")
            query_engine = index.as_query_engine(
                llm=llm,
                streaming=True
            )
            logger.info("Query Engine created successfully.")
            
        except Exception as e:
            raise Exception(f"Error during Query Engine setup: {e}")

        st.success("Data loaded successfully. Ready to answer questions.")
        return query_engine
        
    except Exception as e:
        error_message = f"An error occurred during RAG setup: {e}"
        logger.error(error_message)
        st.error(error_message)
        return None
      


# ----------------------------------------------------------------------------------
st.info("Initializing the AI engine and indexing data. This might take a moment...")

try:
    with st.spinner("⏳ Loading LlamaIndex and GoogleGenAI models. Please wait..."):
        query_engine = setup_rag_engine()
except Exception as e:
 
    st.error("RAG Initialization failed. Please check the terminal for details.")
    query_engine = None 

# ----------------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Resume Assistant - M Safaei",
    layout="wide"
)

st.title("🤖 M. Safaei's AI Career Assistant (Gemini-Powered)")
st.markdown(f"**This is the AI assistant for M Safaei's resume.**")
st.markdown("Ask any question about my skills, projects, and professional background .")
st.divider()

# ----------------------------------------------------------------------------------
if query_engine:
    user_query = st.text_input("Ask a question (e.g., What are your core Python skills? Describe your AI projects.) :")

    if user_query:
        with st.spinner("Searching resume and generating response..."):
            
           
            response = query_engine.query(user_query)
            st.subheader("✅ AI Assistant's Answer:")           
            
            st.info(response)       
        if response.source_nodes:
            st.subheader("📚 Source Context Used:")          
           
            source_text = response.source_nodes[0].get_text()
            st.code(source_text[:500] + "...", language="markdown")           
            
            st.markdown(f"Source file: **{response.source_nodes[0].metadata.get('file_name', 'N/A')}**")
else:
    st.warning("The RAG assistant could not be initialized due to an error. Please check your data folder and API key.")

