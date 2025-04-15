# agentic_rag_bot_openai-api
This Streamlit-based agentic RAG chatbot vectorizes PDF documents and stores them in Qdrant, performs semantic search using OpenAI and Tavily, and leverages CrewAI agents to analyze, rewrite, route, retrieve, and generate responses. Built with Crewai, LangChain, OpenAI, Qdrant, and Tavily technologies.
## 🎥 Demo Video

## 🌟 Features  

- **PDF Upload & Chunking**: Upload PDF files, split them into semantic chunks, and store them as vector embeddings using OpenAI.  
- **Agentic RAG Pipeline (Crewai)**: Implements a multi-agent system with distinct roles: Query Rewriter, Router, Retriever, and Evaluator to enhance information retrieval and response accuracy. 
- **Qdrant Vector Search Integration**: Stores and searches vectorized text chunks using Qdrant, enabling semantic search over uploaded documents.  
- **Tavily Web Search Integration**: Performs real-time web searches when local vector store is not relevant, ensuring up-to-date external information.
- **OpenAI GPT-4o Response Generation**: Generates high-quality, context-aware answers using GPT-4o based solely on retrieved data.
- **Streaming Response**: Displays answers in real-time as they are generated, mimicking ChatGPT-like live response behavior.  
- **First Token Latency Measurement**: Measures and displays how long it takes to get the first response token from OpenAI after query submission.  
- **Collection Management UI**: Create, select, and manage Qdrant collections dynamically via an interactive sidebar.  
- **Built with Streamlit**: Fully interactive UI powered by Streamlit, providing a clean and user-friendly chat experience. 
