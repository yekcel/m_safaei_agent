import streamlit as st
import os
import logging
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.google_genai import GoogleGenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
#os.environ["GEMINI_API_KEY"] = API_KEY_VALUE


@st.cache_resource
def setup_rag_engine():
    """Reads data, indexes it, and prepares the query engine."""    
  
    st.info("Loading resume data and indexing...")
    logger.info("Starting RAG setup process...")
    try:
     
        try:
            logger.info("STEP 1: Attempting to load documents from 'data' folder.")
            documents = SimpleDirectoryReader("data").load_data()
            logger.info(f"Loaded {len(documents)} documents successfully.")
        except FileNotFoundError:
            raise FileNotFoundError("Data folder not found or files are missing.")
        except Exception as e:
            raise Exception(f"Error during document loading: {e}")
        # ----------------------------------------------------------------------------------       
      
       
        # llm = GoogleGenAI(model="gemini-2.5-flash", api_key=API_KEY_VALUE)
        # embed_model = GoogleGenAI(
        #     model="text-embedding-004", 
        #     api_key=API_KEY_VALUE,
        #     embed_model=True          
        # )
       
        # index = VectorStoreIndex.from_documents(documents, llm=llm)
        # embed_model = GoogleGenAI(
        #     model="text-embedding-004",
        #     api_key=API_KEY_VALUE,
        #     embed_model=True         
        # )
        # ----------------------------------------------------------------------------------

      
        # index = VectorStoreIndex.from_documents(
        #     documents, 
        #     llm=llm,             
        #     embed_model=embed_model 
        # )
        try:
            logger.info("STEP 2: Configuring GoogleGenAI models.")
            
           
            llm = GoogleGenAI(model="gemini-2.5-flash", api_key=GEMINI_API_KEY)
            
          
            embed_model = GoogleGenAI(
                model="text-embedding-004", 
                api_key=GEMINI_API_KEY,
                embed_model=True          
            )
            logger.info("LLM and Embedding models configured successfully.")
            
        except Exception as e:
            raise Exception(f"Error during model configuration: {e}")

        # ----------------------------------------------------------------------------------
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
      
        # query_engine = index.as_query_engine(
        #     llm=llm,
        #     streaming=True # For real-time response display
        # )
        
        # st.success("Data loaded successfully. Ready to answer questions.")
        # return query_engine
        # ----------------------------------------------------------------------------------
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

st.title("🤝 AI Assistant for Professional Experiences")
st.markdown(f"**This is the AI assistant for M Safaei's resume.**")
st.markdown("Ask any question about my skills, projects, and professional background below.")
st.divider()

# ----------------------------------------------------------------------------------
if query_engine:
    user_query = st.text_input("Ask a question (e.g., What are your core Python skills? Describe your largest project.):")

    if user_query:
        with st.spinner("Searching resume and generating response..."):
            
           
            response = query_engine.query(user_query)
            
           
            st.subheader("✅ AI Assistant's Answer:")
            
            
            st.info(response.response)

       
        if response.source_nodes:
            st.subheader("📚 Source Context Used:")
            
           
            source_text = response.source_nodes[0].get_text()
            st.code(source_text[:500] + "...", language="markdown")
            
            
            st.markdown(f"Source file: **{response.source_nodes[0].metadata.get('file_name', 'N/A')}**")
else:
    st.warning("The RAG assistant could not be initialized due to an error. Please check your data folder and API key.")

