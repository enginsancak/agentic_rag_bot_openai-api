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


## 🚀 Quick Start
### Prerequisites
- Python 3.8 or higher
- OpenAI API key
- Tavily API key
- Qdrant URL
- Qdrant API key
- Enter the API keys and the Qdrant URL into the corresponding fields in the .env file
  
### Installation

**1️. Clone the Repository:**

      git clone https://github.com/enginsancak/agentic_rag_bot_openai-api

      cd agentic_rag_bot_openai-api

**2. Create and activate a virtual environment:**
   
      python -m venv venv

      source venv/bin/activate  # On Windows: .\venv\Scripts\activate

**3. Install dependencies:**

      pip install -r requirements.txt

### Running the Application

**1. Start the Streamlit app:**

      streamlit run main.py

**2. Open your browser and navigate to http://localhost:8501**

## 📁 Project Structure

     agentic_rag_bot_openai-api/
     ├── main.py                             # Main application file
     ├── .env                                # Environment variables (API keys, Qdrant URL)
     ├── requirements.txt                    # Project dependencies
     ├── README.md                           # Project documentation
     





