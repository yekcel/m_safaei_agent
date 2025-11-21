import streamlit as st
import os
import logging
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.google_genai import GoogleGenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FALLBACK_API_KEY = "AIzaSyCr4ND6SOcAF8UsptOld7VD0XjGyTtSqYk"
API_KEY_VALUE = os.getenv("GEMINI_API_KEY", FALLBACK_API_KEY)
# 1. Configuration and Model Setup
# Ensure the GEMINI_API_KEY environment variable is set in your deployment environment (e.g., Streamlit Cloud Secrets)
#os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["GEMINI_API_KEY"] = API_KEY_VALUE
# --- RAG Logic ---

@st.cache_resource
def setup_rag_engine():
    """Reads data, indexes it, and prepares the query engine."""
    
    # Display loading message in the UI
    st.info("Loading resume data and indexing...")
    logger.info("Starting RAG setup process...")
    try:
        # A: Load documents from the 'data' folder
        # Make sure your data files (txt, md, pdf) are in a folder named 'data'
        try:
            logger.info("STEP 1: Attempting to load documents from 'data' folder.")
            documents = SimpleDirectoryReader("data").load_data()
            logger.info(f"Loaded {len(documents)} documents successfully.")
        except FileNotFoundError:
            raise FileNotFoundError("Data folder not found or files are missing.")
        except Exception as e:
            raise Exception(f"Error during document loading: {e}")
        # ----------------------------------------------------------------------------------
        # B: Configure Models (LLM for Generation and Embed Model for Indexing)
        global API_KEY_VALUE
        # if API_KEY_VALUE == "AIzaSyCr4ND6SOcAF8UsptOld7VD0XjGyTtSqYk":
        #     st.error("Please replace 'AIzaSyCr4ND6SOcAF8UsptOld7VD0XjGyTtSqYk' with your actual key in app.py!")
        #     return None
        # B: Configure the Gemini Model
        # llm = GoogleGenAI(model="gemini-2.5-flash", api_key=API_KEY_VALUE)
        # embed_model = GoogleGenAI(
        #     model="text-embedding-004", 
        #     api_key=API_KEY_VALUE,
        #     embed_model=True          
        # )
        # C: Build the Index (LlamaIndex automatically handles Chunking and Embedding)
        # index = VectorStoreIndex.from_documents(documents, llm=llm)
        # embed_model = GoogleGenAI(
        #     model="text-embedding-004", # مدل جدید Google AI برای جاسازی
        #     api_key=API_KEY_VALUE,
        #     embed_model=True          # مشخص کردن که این مدل برای جاسازی است
        # )
        # ----------------------------------------------------------------------------------

        # C: ساخت ایندکس (Indexing) - ارسال صریح embed_model
        # index = VectorStoreIndex.from_documents(
        #     documents, 
        #     llm=llm,              # مدل برای پاسخ دهی
        #     embed_model=embed_model # مدل برای جاسازی (Embedding)
        # )
        try:
            logger.info("STEP 2: Configuring GoogleGenAI models.")
            
            # مدل تولید پاسخ (Generation LLM)
            llm = GoogleGenAI(model="gemini-2.5-flash", api_key=API_KEY_VALUE)
            
            # مدل جاسازی (Embedding Model)
            embed_model = GoogleGenAI(
                model="text-embedding-004", 
                api_key=API_KEY_VALUE,
                embed_model=True          
            )
            logger.info("LLM and Embedding models configured successfully.")
            
        except Exception as e:
            raise Exception(f"Error during model configuration: {e}")

        # --- مرحله ۳: ساخت ایندکس (Indexing) ---
        try:
            logger.info("STEP 3: Starting VectorStoreIndex creation (Chunking and Embedding).")
            # این خط، نقطه اصلی مصرف منابع و احتمال خطا است
            index = VectorStoreIndex.from_documents(
                documents, 
                llm=llm,              
                embed_model=embed_model 
            )
            logger.info("VectorStoreIndex created successfully.")
            
        except Exception as e:
            raise Exception(f"Error during Indexing/Embedding: {e}")
        # D: Create the Query Engine
        # query_engine = index.as_query_engine(
        #     llm=llm,
        #     streaming=True # For real-time response display
        # )
        
        # st.success("Data loaded successfully. Ready to answer questions.")
        # return query_engine
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
        
    except FileNotFoundError:
        st.error("Error: The 'data' folder or resume files were not found. Please ensure your project structure is correct.")
        return None
    except Exception as e:
        st.error(f"An error occurred during RAG setup: {e}")
        return None

# Attempt to set up the RAG engine
#query_engine = setup_rag_engine()
# Initialize the RAG engine با نمایشگر انتظار
st.info("Initializing the AI engine and indexing data. This might take a moment...")

try:
    with st.spinner("⏳ Loading LlamaIndex and GoogleGenAI models. Please wait..."):
        query_engine = setup_rag_engine()
except Exception as e:
    # خطا قبلاً در setup_rag_engine لاگ شده، اما اینجا برای اطمینان مجدد نمایش داده می شود
    st.error("RAG Initialization failed. Please check the terminal for details.")
    query_engine = None # مطمئن می شویم که موتور پرس و جو None باشد

# --- Streamlit UI ---

# 2. Page Configuration and Title
st.set_page_config(
    page_title="AI Resume Assistant - M Safaei",
    layout="wide"
)

st.title("🤝 AI Assistant for Professional Experiences")
# >>> CUSTOMIZE THIS LINE: Put your name here
st.markdown(f"**This is the AI assistant for M Safaei's resume.**")
st.markdown("Ask any question about my skills, projects, and professional background below.")

# Horizontal separator
st.divider()

# 3. User Input
if query_engine: # Only show input if RAG engine was successfully initialized
    user_query = st.text_input("Ask a question (e.g., What are your core Python skills? Describe your largest project.):")

    if user_query:
        with st.spinner("Searching resume and generating response..."):
            
            # 4. Execute the RAG Query Engine
            response = query_engine.query(user_query)
            
            # 5. Display the Response
            st.subheader("✅ AI Assistant's Answer:")
            
            # Display the streaming response
            st.info(response.response)

        # (Optional) Display the retrieved source context
        if response.source_nodes:
            st.subheader("📚 Source Context Used:")
            
            # Get the first retrieved text chunk
            source_text = response.source_nodes[0].get_text()
            st.code(source_text[:500] + "...", language="markdown")
            
            # Display the filename used
            st.markdown(f"Source file: **{response.source_nodes[0].metadata.get('file_name', 'N/A')}**")
else:
    st.warning("The RAG assistant could not be initialized due to an error. Please check your data folder and API key.")

