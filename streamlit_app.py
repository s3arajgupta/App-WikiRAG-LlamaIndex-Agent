"""Streamlit Web Application for WikiRAG: Autonomous ReAct Wikipedia Chatbot."""

import os
import streamlit as st
from src.wikirag import (
    create_index,
    create_react_agent,
    run_agent,
    SUPPORTED_MODELS,
    WikiRAGConfig,
    get_api_key
)

# Page configuration
st.set_page_config(
    page_title="WikiRAG — Autonomous Wikipedia Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .badge-success { background-color: #DEF7EC; color: #03543F; }
    .badge-warning { background-color: #FEF08A; color: #854D0E; }
    .badge-info { background-color: #E0E7FF; color: #3730A3; }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "index" not in st.session_state:
    st.session_state.index = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "indexed_pages" not in st.session_state:
    st.session_state.indexed_pages = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Controls
with st.sidebar:
    st.image("https://raw.githubusercontent.com/run-llama/llama_index/main/docs/static/img/llama.png", width=64)
    st.title("Knowledge Engine")

    st.subheader("1. LLM & Runtime")
    model_choice = st.selectbox(
        "Choose Inference Model:",
        options=SUPPORTED_MODELS,
        index=0 if get_api_key() else 3,
        help="Select 'mock-mode' to test indexing and ReAct reasoning without an OpenAI API key."
    )

    api_key_input = st.text_input(
        "OpenAI API Key:",
        value=get_api_key(),
        type="password",
        help="Leave blank to use environment variable or test in mock-mode."
    )
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input

    st.divider()

    st.subheader("2. Wikipedia Indexing")
    preset_choice = st.selectbox(
        "Load Topic Preset:",
        ["Custom Query", "Batman, Paris", "Python, Artificial Intelligence", "Star Wars, Cinema"]
    )
    default_query = "Paris, Batman" if preset_choice == "Custom Query" else preset_choice
    pages_query = st.text_input(
        "Pages to Index (comma-separated):",
        value=default_query,
        placeholder="e.g. Paris, Batman, Python"
    )

    with st.expander("Advanced RAG Settings"):
        chunk_size = st.slider("Chunk Size (tokens)", min_value=100, max_value=1000, value=256, step=50)
        chunk_overlap = st.slider("Chunk Overlap (tokens)", min_value=0, max_value=100, value=32, step=8)
        similarity_k = st.slider("Top-K Passages", min_value=1, max_value=10, value=4)

    if st.button("🔨 Build Knowledge Index", type="primary", use_container_width=True):
        if not pages_query.strip():
            st.error("Please enter at least one Wikipedia page title.")
        else:
            with st.spinner(f"Fetching and indexing '{pages_query}'..."):
                try:
                    cfg = WikiRAGConfig(
                        model_name=model_choice,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        similarity_top_k=similarity_k
                    )
                    st.session_state.index = create_index(pages_query, model_name=model_choice, config=cfg)
                    st.session_state.agent = create_react_agent(model_choice, st.session_state.index, config=cfg)
                    st.session_state.indexed_pages = pages_query
                    st.success(f"Indexed: {pages_query}")
                except Exception as e:
                    st.error(f"Indexing error: {e}")

    if st.session_state.indexed_pages:
        st.info(f"Active Knowledge: **{st.session_state.indexed_pages}**")

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main Header
st.markdown("<div class='main-header'>WikiRAG: Autonomous ReAct Wikipedia Agent</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>Dynamic VectorStore indexing with multi-hop ReAct (Reasoning + Acting) retrieval on Wikipedia.</div>",
    unsafe_allow_html=True
)

# Status Indicator
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.session_state.index:
        st.markdown("<span class='status-badge badge-success'>Index Active</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='status-badge badge-warning'>Index Not Built</span>", unsafe_allow_html=True)
with col2:
    if model_choice == "mock-mode":
        st.markdown("<span class='status-badge badge-info'>Offline Mock Mode</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span class='status-badge badge-info'>Live: {model_choice}</span>", unsafe_allow_html=True)

st.write("")

# Display Conversation History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # If response contains reasoning trace, display in expander
        content = msg["content"]
        if "[ReAct Reasoning Trace]" in content:
            parts = content.split("> Final Answer:")
            trace = parts[0].strip()
            final_answer = parts[1].strip() if len(parts) > 1 else content
            st.markdown(final_answer)
            with st.expander("Inspect ReAct Reasoning Trace"):
                st.code(trace, language="markdown")
        else:
            st.markdown(content)

# User Chat Input
if prompt := st.chat_input("Ask a question about the indexed Wikipedia topics..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if agent is ready
    if not st.session_state.agent:
        # Auto-initialize in mock mode or notify
        if not st.session_state.index:
            with st.spinner("Auto-indexing default pages ('Paris, Batman')..."):
                cfg = WikiRAGConfig(model_name=model_choice)
                st.session_state.index = create_index("Paris, Batman", model_name=model_choice, config=cfg)
                st.session_state.agent = create_react_agent(model_choice, st.session_state.index, config=cfg)
                st.session_state.indexed_pages = "Paris, Batman"

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("ReAct Agent Reasoning..."):
            response = run_agent(st.session_state.agent, prompt)
            if "[ReAct Reasoning Trace]" in response:
                parts = response.split("> Final Answer:")
                trace = parts[0].strip()
                final_answer = parts[1].strip() if len(parts) > 1 else response
                st.markdown(final_answer)
                with st.expander("Inspect ReAct Reasoning Trace"):
                    st.code(trace, language="markdown")
            else:
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})