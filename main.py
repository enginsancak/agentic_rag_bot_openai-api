import streamlit as st
import time
import os
import openai
import uuid
import pdfplumber
from dotenv import load_dotenv
from qdrant_client import QdrantClient  
from qdrant_client.models import Distance, VectorParams, SearchParams, PointStruct
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults
from crewai import Crew, Agent, Task
from crewai.tools import tool
from crewai.tools import BaseTool
from typing import Type, Any
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")


client = openai.OpenAI()
qdrant_client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
    timeout=60)


# ----------------------------------------
# 🔧 FUNCTIONS
# ----------------------------------------
def load_pdf(pdf_file):
    """Reads the PDF file page by page and returns only the plain text."""
    full_text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    return full_text


def chunk_text(text, chunk_size=1000, overlap=200):
    """Splits the entire text into chunks. No metadata like page number is added."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_text(text)
    return [{"text": chunk} for chunk in chunks]



def get_embedding(text):
    """Embeds the text using OpenAI."""
    
    if not text or not isinstance(text, str) or text.strip() == "":
        print("⚠️ ERROR: An invalid text was provided to the get_embedding function.")
        return None
    
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            dimensions=1536,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"🚨 OpenAI Embedding API Error: {e}")
        return None
    

def save_to_qdrant(chunks):
    """Converts the list of text chunks into embeddings and stores them in Qdrant."""
    collection = st.session_state.get("collection_name", None)
    if not collection:
        st.warning("⚠️ Please create and select a Qdrant collection first.")
        return

    points = []
    for chunk in chunks:
        text = chunk["text"]
        embedding = get_embedding(text)
        if embedding:
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={"text": text}
            ))

    if points:
        qdrant_client.upsert(collection_name=collection, points=points)
        print(f"✅ {len(points)} chunks have been uploaded to Qdrant. Collection: {collection}")


# ----------------------------------------
# 🧰 TOOLS
# ----------------------------------------
class TavilySearchInput(BaseModel):
    """The query required to perform a web search using the Tavily API."""
    query: str = Field(..., description="The search query for Tavily Web Search")

class TavilySearchTool(BaseTool):
    name: str = "Tavily Web Search"
    description: str = "Uses Tavily API to perform a web search and return relevant results."
    args_schema: Type[BaseModel] = TavilySearchInput

    def _run(self, query: str) -> Any:
        search = TavilySearchResults()
        return search.run(query)


@tool("Qdrant Similarity Retriever")
def retrieve_highest_similarity(query: str) -> float:
    """
    Retrieves the highest similarity score from Qdrant based on the input query.
    Converts the query to a vector using OpenAI embeddings and searches the Qdrant collection.
    Returns the highest similarity score among the top 5 results.
    """
    collection = st.session_state.get("collection_name", None)
    if not collection:
        return ["⚠️ No collection selected. Please create and select one before searching."]
    
    query_vector = embedding_model.embed_query(query)

    search_results = qdrant_client.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=4
    )

    if not search_results:
        print("No results found.")
        return 0.0

    best_match = max(search_results, key=lambda x: x.score)
    print(f"Highest similarity score: {best_match.score}")
    return best_match.score


@tool("Qdrant Vector Search")
def search_qdrant_tool(query: str) -> list:
    """Fetches the most relevant text chunks from Qdrant vector database based on semantic similarity with the input query."""
    collection = st.session_state.get("collection_name", None)
    if not collection:
        return ["⚠️ No collection selected. Please create and select one before searching."]
    query_vector = embedding_model.embed_query(query)
    search_result = qdrant_client.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=4,
        search_params=SearchParams(hnsw_ef=64, exact=False)
    )
    relevant_chunks = [hit.payload["text"] for hit in search_result]
    return relevant_chunks


web_search_tool = TavilySearchTool()
retrieve_highest_similarity_tool = retrieve_highest_similarity
vector_search_tool = search_qdrant_tool


# ----------------------------------------
# 🤖 AGENTS & TASKS
# ----------------------------------------

# Rewrite Agent
rewrite_agent = Agent(
    role="Query Expansion Specialist",
    goal="Enhance and expand the search query to retrieve more relevant results",
    backstory="""An expert in search query optimization, NLP, and semantic expansion.
    Uses advanced techniques like synonym enrichment, contextual expansion, and 
    domain-specific augmentation to improve search effectiveness.""",
    llm=llm,
    allow_delegation=False,
    verbose=True)

# Router Agent
router_agent = Agent(
    role="Traffic Router",
    goal="Determine the appropriate search destination based on query content",
    backstory="""Specialized in query analysis and routing with deep understanding
    of various data sources and their applicability.""",
    llm=llm,
    allow_delegation=False,
    verbose=True)

# Retriever Agent
retriever_agent = Agent(
    role="Information Retriever",
    goal="Fetch relevant information from appropriate source",
    backstory="""Experienced in efficient information retrieval from multiple
    sources with expertise in both web and vector store searches.""",
    llm=llm,
    allow_delegation=False,
    verbose=True)


# Evaluator Agent
evaluator_agent = Agent(
    role="Content Evaluator",
    goal="Assess the relevance and completeness of retrieved information",
    backstory="""Expert in content analysis and quality assessment with
    strong analytical skills and attention to detail.""",
    llm=llm,
    allow_delegation=False,
    verbose=True)


# Task for Rewrite Agent
rewrite_task = Task(
    description="""Analyze the input query: {query}, and IF necessary expand it to improve search effectiveness.
    Use techniques like:
    - **Synonym Expansion:** Include synonyms or related terms to broaden the query scope.
    - **Contextual Expansion:** Add relevant context based on the query's meaning.
    - **Domain-Specific Expansion:** If the query is technical, include related industry terms.
    - **Entity Recognition:** Identify and expand named entities (e.g., "AI" → "Artificial Intelligence, Machine Learning, Deep Learning").
    
    Ensure that the query remains natural, relevant, and does not introduce unrelated concepts.
    Finish in 2 sentences.
    Ask a question in both sentences.
    """,
    expected_output="""An expanded query that maintains the original intent but includes additional
    relevant terms to improve search results. The final query should be well-formed and readable.""",
    agent=rewrite_agent)


# Task for Router Agent
router_task = Task(
    description="""Analyze the rewritten query to determine if it's related to Qdrant vectorstore search.
    Return 'vector_store' if the query is related to Qdrant, otherwise return 'web_search'.
    Find the highest similarity score by using the given tool and check whether this score is greater than 0.45 or not.
    If the similarity score is greater than 0.45, output 'vector_store'; otherwise, output 'web_search'.""",
    expected_output="""A string containing either 'vector_store' or 'web_search' based on
    the query analysis. The output should be lowercase and match exactly one of these two options.""",
    agent=router_agent,
    tools=[retrieve_highest_similarity_tool],
    context=[rewrite_task])


# Task for Retriever Agent
retriever_task = Task(
    description="""Use the appropriate search tool based on the router's output to fetch relevant information.
    For 'vector_store', use vector_search_tool. For 'web_search', use web_search_tool.""",
    expected_output="""A long and detailed string containing the search results. The output should include relevant information that directly addresses the rewritten query, with sources when available.""",
    agent=retriever_agent,
    tools=[vector_search_tool, web_search_tool],
    context=[rewrite_task, router_task])


# Task for Evaluator Agent
evaluator_task = Task(
    description="""Analyze the retrieved information to determine if it's relevant to the query,
    and contains a sufficient answer. Consider factors like content relevance, completeness,
    and reliability of the information.""",
    expected_output="""A string containing 'yes' if the content is relevant and contains
    a sufficient answer, or 'no' if the content is irrelevant or insufficient.""",
    agent=evaluator_agent,
    context=[rewrite_task, retriever_task])



# ----------------------------------------
# 🧠 LLM 
# ----------------------------------------
def stream_answer_from_openai(rewritten, retrieved, evaluation):
    prompt = f"""Query: {rewritten}

Retrieved Info:\n{retrieved}

Evaluator Decision: {evaluation}

Only use the retrieved information to give a long and detailed answer. Do not invent anything."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": """Generate a comprehensive and well-structured answer to the query based on the retrieved information.
            Only respond based on the retrieved information. DO NOT add any external information
            If the evaluator returned 'yes', synthesize the information into a clear, logical, and concise response that directly addresses the query.
            If 'no', return 'Unable to answer the given query' without adding unnecessary information.
            Display the sources at the end of the output.
            Ensure responses maintain coherence, factual accuracy, and avoid speculative statements.
            **Give a long and detailed answer.** """},
            {"role": "user", "content": prompt}
        ],
        stream=True
    )
    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# ----------------------------------------
# 🧭 MAIN EXECUTION FLOW
# ----------------------------------------

st.set_page_config(page_title="Agentic RAG Chatbot", layout="wide")

# Retrieve Available Qdrant Collections And Store Them In Session State
if "collections" not in st.session_state:
    try:
        existing_collections = qdrant_client.get_collections().collections
        st.session_state.collections = [c.name for c in existing_collections]
    except Exception as e:
        st.error(f"Failed to retrieve collections: {e}")
        st.session_state.collections = []

if "collection_name" not in st.session_state:
    st.session_state.collection_name = None


# Sidebar Panel For General Settings
with st.sidebar:
    st.title("⚙️ Settings & Information")
    query_time_placeholder = st.empty()  
    st.markdown("---")  

    # Manage And Select Qdrant Vector Collections
    st.title("🧠 Qdrant Collections")

    if "collections" not in st.session_state:
        st.session_state.collections = []
    if "collection_name" not in st.session_state:
        st.session_state.collection_name = None

    new_collection_name = st.text_input("New Collection Name", placeholder="example: law_pdfs")

    if st.button("📁 Create Collection"):
        if new_collection_name and new_collection_name not in st.session_state.collections:
            try:
                qdrant_client.create_collection(
                    collection_name=new_collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
                )
                if new_collection_name not in st.session_state.collections:
                    st.session_state.collections.append(new_collection_name)
                st.session_state.collection_name = new_collection_name
                st.success(f"✅ '{new_collection_name}' collection has been created and selected.")
            except Exception as e:
                if "already exists" in str(e):
                    st.warning(f"⚠️ '{new_collection_name}' collection already exists. You can select it from below.")
                else:
                    st.error(f"Error: {e}")

    for name in st.session_state.collections:
        if st.button(name, key=name):
            st.session_state.collection_name = name

    if st.session_state.collection_name:
        st.sidebar.success(f"🔗 Active Collection: {st.session_state.collection_name}")
    else:
        st.sidebar.warning("❗No active collection has been selected yet.")

    # Upload Pdf Files And İndex Them Into Qdrant As Vector Chunks
    st.title("📎 PDF Upload and Indexing")
    uploaded_pdfs = st.file_uploader("Upload your PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded_pdfs:
        if st.button("📚 Chunk & Index"):
            all_chunks = []
            for uploaded_file in uploaded_pdfs:
                pdf_name = uploaded_file.name
                st.write(f"🔍 `{pdf_name}` processing...")
                full_text = load_pdf(uploaded_file)
                chunks = chunk_text(full_text)
                all_chunks.extend(chunks)
            save_to_qdrant(all_chunks)
            st.success(f"✅ Total {len(all_chunks)} amounts of chunks uploaded to Qdrant.")


# Load And Display Previous Chat Messages From Session State To Maintain Conversation History
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Process User Input Through a Multi-Step Agent Pipeline
st.title("💬 Agentic RAG Chatbot")
user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    left_col, right_col = st.columns([3, 1])
    with left_col:
        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            query_time_placeholder = st.empty()
            start_time = time.time()

            # 1. Rewrite step
            status_placeholder.info("✏️ Rewriting the user query...")
            rewrite_crew = Crew(agents=[rewrite_agent], tasks=[rewrite_task], verbose=True)
            rewritten_result = rewrite_crew.kickoff(inputs={"query": user_input})
            rewritten = str(rewritten_result)

            # 2. Router step
            status_placeholder.info("📍 Routing...")
            router_crew = Crew(agents=[router_agent], tasks=[router_task], verbose=True)
            routing_result = router_crew.kickoff(inputs={"query": rewritten})
            routing = str(routing_result).strip().lower()

            # 3. Retriever step
            status_placeholder.info("📦 Retrieving the related data...")
            retriever_crew = Crew(agents=[retriever_agent], tasks=[retriever_task], verbose=True)
            retrieved_result = retriever_crew.kickoff(inputs={"query": rewritten, "routing": routing})
            retrieved = str(retrieved_result)

            # 4. Evaluator step
            status_placeholder.info("🧠 Evaluating results...")
            evaluator_crew = Crew(agents=[evaluator_agent], tasks=[evaluator_task], verbose=True)
            evaluation_result = evaluator_crew.kickoff(inputs={"query": rewritten, "retrieved": retrieved})
            evaluation = str(evaluation_result).strip().lower()

            # 5. Answer step (streaming)
            status_placeholder.info("💬 Streaming answer...")
            stream_area = st.empty()
            streamed_text = ""
            first_token_received = False

            for token in stream_answer_from_openai(rewritten, retrieved, evaluation):
                if not first_token_received:
                    first_token_received = True
                    first_token_time = time.time()
                    query_time_placeholder.write(f"⏳ **First Token Latency:** {first_token_time - start_time:.2f} seconds")
                streamed_text += token
                stream_area.markdown(streamed_text + "▌")

            stream_area.markdown(streamed_text)
            status_placeholder.success("✅ Answer ready!")
            st.session_state.messages.append({"role": "assistant", "content": streamed_text})

    with right_col:
        with st.expander("🧠 Agent Insights", expanded=True):
            st.markdown(f"📝 **Rewritten Query:**\n\n`{rewritten}`")
            st.markdown(f"📍 **Routing Decision:** `{routing}`")
            st.markdown("📦 **Retriever Output:**")
            st.markdown(f"```text\n{retrieved}\n```")
            st.markdown(f"🧠 **Evaluator Decision:** `{evaluation}`")






