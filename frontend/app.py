import os
import streamlit as st
import requests
import json
from predictionguard import PredictionGuard
import time

# Initialize PredictionGuard client
client = PredictionGuard(url=os.getenv("PREDICTIONGUARD_URL","https://api.predictionguard.com"))

# Configure the page
st.set_page_config(
    page_title="Deep Research Assistant",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stTextInput>div>div>input {
        font-size: 18px;
    }
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        font-size: 18px;
        padding: 0.5rem 1rem;
    }
    .raw-results {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🔬 Deep Research Assistant")
st.markdown("""
    Ask any biomedical research question and get comprehensive answers from multiple sources including:
    - PubMed articles
    - Clinical trials
    - BioRxiv preprints
""")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to format the response using PG LLM
def format_response(query: str, raw_results: list) -> str:
    system_prompt = """You are a biomedical research assistant. Your task is to synthesize information from multiple sources into a clear, well-structured response.
    Format the response in a way that:
    1. Directly answers the user's question
    2. Provides relevant citations and sources
    3. Highlights key findings and implications
    4. Uses clear, professional language
    
    The response should be well-organized and easy to read."""

    # Prepare the context from raw results
    context = json.dumps(raw_results, indent=2)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {query}\n\nSources: {context}\n\nPlease provide a comprehensive answer based on these sources."}
    ]

    try:
        result = client.chat.completions.create(
            model=os.getenv("PREDICTIONGUARD_MODEL","Hermes-3-Llama-3.1-70B"),
            messages=messages,
            max_completion_tokens=2000,
            temperature=0.1
        )
        return result['choices'][0]['message']['content']
    except Exception as e:
        return f"Error formatting response: {str(e)}"

# Function to make API call to our FastAPI backend
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8080")
def get_research_results(query: str, max_results: int = 10):
    try:
        response = requests.post(
            f"{BACKEND_API_URL}/deepresearch",
            json={"query": query, "max_results": max_results}
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# Create the input area
query = st.text_input("Enter your research question:", placeholder="e.g., What are the latest developments in mRNA vaccine technology?")

# Add a slider for max results
max_results = st.slider("Maximum results per source:", min_value=5, max_value=20, value=10)

# Create columns for the submit button
col1, col2, col3 = st.columns([1,2,1])
with col2:
    submit_button = st.button("🔍 Search", use_container_width=True)

# Process the query when submitted
if submit_button and query:
    with st.spinner("Searching multiple sources..."):
        # Get raw results from the API
        raw_results = get_research_results(query, max_results)
        
        # Format the response using PG LLM
        formatted_response = format_response(query, raw_results)
        
        # Add to chat history
        st.session_state.chat_history.append({
            "query": query,
            "response": formatted_response,
            "raw_results": raw_results
        })

# Display chat history
for i, chat in enumerate(reversed(st.session_state.chat_history)):
    st.markdown("---")
    st.markdown(f"### Q: {chat['query']}")
    st.markdown(chat['response'])
    
    # Add a toggle for raw results
    show_raw = st.checkbox("Show Raw Results", key=f"show_raw_{i}")
    if show_raw:
        st.markdown('<div class="raw-results">', unsafe_allow_html=True)
        st.json(chat['raw_results'])
        st.markdown('</div>', unsafe_allow_html=True)
