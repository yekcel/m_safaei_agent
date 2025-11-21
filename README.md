# 🤖 M. Safaei's AI Career Assistant (Gemini RAG Demo)

[](https://msafaeiagent.streamlit.app/)

## 🚀 Live Demo

**Test the fully-deployed application here:**
👉 **[Live AI Career Assistant (Streamlit Cloud)](https://msafaeiagent.streamlit.app/)**

-----

## 🎯 Overview

This repository contains the source code for a **Retrieval-Augmented Generation (RAG) Agent** designed to answer complex, personalized questions about my 15+ years of professional experience, projects, and skills.

This project serves as a practical demonstration of **production-grade AI Engineering, LLM Orchestration, and robust cloud deployment** techniques.

## ✨ Key Features & Technical Details

  * **LLM Integration:** Utilizes the **Google Gemini 2.5-Flash** model for high-quality, real-time response generation.
  * **RAG Orchestration:** Built with **LlamaIndex** to intelligently retrieve context from unstructured resume data, ensuring grounded and relevant answers.
  * **Deployment Stability:** Features a highly resilient deployment pipeline, including:
      * **Custom Chunking Logic:** Implemented using `SentenceSplitter` to optimize context size and stability.
      * **Streamlit Caching (`@st.cache_resource`):** Ensures the heavy indexing process is only performed once, preventing repeated API calls and deployment failures.
      * Successfully resolved complex dependency conflicts and **API 403 Permission Denied** errors related to cloud deployment.
  * **Full-Stack Presentation:** Deployed live on **Streamlit Cloud** for a user-friendly, interactive Q\&A experience.

## ⚙️ Technology Stack

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **LLM/API** | Google Gemini (2.5-Flash) | Core Language Model for Generation |
| **RAG Framework** | LlamaIndex | Data Indexing, Retrieval, and Query Engine |
| **Deployment** | Streamlit Cloud | Frontend and Application Hosting |
| **Language** | Python | Core Development Language |



## 👤 Author

**Masan Safaei** (M. Safaei)

  * **LinkedIn:** [https://www.linkedin.com/in/safaei-ai/]
  * **Email:** `mh.safaei@gmail.com`
