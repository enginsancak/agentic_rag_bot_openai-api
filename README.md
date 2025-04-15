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
     

## 💡 Usage Guide

**1. Setup**
- Enter the API keys and the Qdrant URL into the corresponding fields in the .env file
- Run the code in the terminal
- Create a collection or select an existing one to load the embeddings

**2. Document Upload**
- Upload a PDF document
- Click the "Chunk & Index" button to save the embeddings to the Qdrant vector database.

**3.Chatting**
- Ask questions
- Wait and view the Answer
- View all agent outputs in the right panel
- If the query is related to the documents in the vector database, the answer is retrieved from there; otherwise, a web search is used to generate the response

## 🔧 Available Models
- **openai gpt-4o**

## 🔍 Technical Details
**Components**
- **Frontend:** Streamlit
- **Embeddings:** OpenAI Embeddings (text-embedding-3-small)
- **LLM Provider:** OpenAI GPT-4o
- **PDF Processing:** PDFPlumber
- **Text Splitting:** RecursiveCharacterTextSplitter
- **Vector Store:** Qdrant
- **Web Search:** Tavily API
- **Agent Framework:** CrewAI

**Process Flow**

**1. Environment Setup & Credential Loading**
- API keys and connection URLs (OpenAI, Tavily, Qdrant) are loaded from the .env file.
- Required clients (OpenAI, QdrantClient) are initialized.

**2. Collection Management**
- The user either creates a new Qdrant collection or selects an existing one via the Streamlit sidebar.
- Collections are retrieved using qdrant_client.get_collections() and displayed as buttons.
- An active collection must be selected before uploading or retrieving data.

**3. Document Upload**
- The user uploads one or more PDF files via the UI.

**4. PDF Processing**
- Uploaded PDF files are read page-by-page using pdfplumber.
- All text is merged and converted into clean plain text.

**5. Text Chunking**
- The full text is split into chunks using RecursiveCharacterTextSplitter.
- Chunks are overlapping and ~1000 characters in length.

**6. Text Embedding**
- Each chunk is converted into a 1536-dimensional vector using OpenAIEmbeddings (text-embedding-3-small).

**7. Vector Storage in Qdrant**
- Generated vectors are stored in the selected collection using QdrantClient.upsert().
- Each vector is stored with a payload containing the original text.

**8. User Query Submission**
- The user submits a search query.
- The query is logged into the chat history and the flow begins.

**9. Query Rewriting (Agent 1)**
- The Rewrite Agent semantically enhances the query.
- Goal: Optimize the query for more effective retrieval.

**10. Routing Decision (Agent 2)**
- The Router Agent performs a similarity search on Qdrant using the rewritten query.
- Based on the highest similarity score:
- Score > 0.45 → use vector_store; otherwise → use web_search.

**11. Information Retrieval (Agent 3)**
- The Retriever Agent fetches information from the selected source (Qdrant or Tavily API):

If vector_store → top 4 matching chunks are retrieved.

If web_search → Tavily API is used.

**12. Information Evaluation (Agent 4)**
- The Evaluator Agent assesses whether the retrieved information is relevant and sufficient.
- Decision: yes (sufficient) or no (insufficient).

**13. Answer Generation with LLM**
- ChatOpenAI (gpt-4o) generates a response only using the retrieved content.
- If Evaluator response is "no" → returns “Unable to answer the given query.”

**14. Streaming Response to UI**
- The generated answer is streamed to the user interface in real time.
- First token latency is measured and shown.

**15. Agent Insights Display**
- Each agent’s output (rewrite, route, retrieve, evaluate) is displayed for transparency.
- The user can track each stage of the reasoning.

**16. Session Persistence**
- Queries and responses are stored in st.session_state to preserve conversation history.



