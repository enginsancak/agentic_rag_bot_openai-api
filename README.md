# agentic_rag_bot_openai-api
This Streamlit-based agentic RAG chatbot vectorizes PDF documents and stores them in Qdrant, performs semantic search using OpenAI and Tavily, and leverages CrewAI agents to analyze, rewrite, route, retrieve, and generate responses. Built with Crewai, LangChain, OpenAI, Qdrant, and Tavily technologies.
## 🎥 Demo Video

## 🌟 Features  

- **PDF Upload & Processing**: Upload PDF files and extract meaningful text for AI-driven analysis.  
- **RAG-Based Retrieval**: Uses Retrieval-Augmented Generation (RAG) to provide accurate and contextual responses.  
- **Vector-Based Semantic Search**: Stores and retrieves document chunks using an In-Memory VectorStore.  
- **Streaming AI Responses**: Get real-time, token-by-token responses with GROQ models.
- **Asynchronous API Calls**: GROQ API is executed asynchronously for faster response times and improved efficiency.
- **Step-by-Step Query Breakdown**: AI explains its reasoning in multiple structured steps before answering.  
- **Performance Monitoring**: Displays response time, chunk count, and PDF processing duration.  
- **Advanced Model Settings**: Customize temperature, max tokens, chunk size, and overlap via UI controls.  
- **Relevant Chunk Display:** Displays the most relevant document chunks in the right panel when a query is made, improving context awareness and transparency. 
- **Clean & Interactive UI**: Built with Streamlit for a user-friendly and responsive interface.
