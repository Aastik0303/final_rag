
import streamlit as st
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import tempfile
import base64
import io
from datetime import datetime

# Image Processing
from PIL import Image

# LangChain Core & Google
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Modern Agent
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# RAG
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# YouTube processing
import re
import urllib.request
from urllib.parse import urlparse, parse_qs
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    TRANSCRIPT_AVAILABLE = True
except ImportError:
    TRANSCRIPT_AVAILABLE = False
try:
    import yt_dlp
    YTDLP_AVAILABLE = True
except ImportError:
    YTDLP_AVAILABLE = False

# ═════════════════════════════════════════════
# CUSTOM CSS
# ═════════════════════════════════════════════

def load_custom_css():
    st.markdown("""
    <style>
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%); }
    [data-testid="stSidebar"] * { color: white !important; }
    .stApp > header { background-color: transparent; }
    .stChatMessage {
        background-color: rgba(255,255,255,0.95);
        border-radius: 15px; padding: 15px; margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1, h2, h3 { color: white !important; font-weight: 700 !important; }
    .stTextInput > div > div > input { border-radius: 10px; border: 2px solid #667eea; }
    .stButton > button {
        border-radius: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white; font-weight: 600; border: none;
        padding: 10px 24px; transition: all 0.3s ease;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 12px rgba(0,0,0,0.2); }
    .stFileUploader { background-color: rgba(255,255,255,0.1); border-radius: 10px; padding: 15px; }
    [data-testid="stMetricValue"] { font-size: 28px; color: #667eea; font-weight: 700; }
    .stAlert { border-radius: 10px; border-left: 5px solid #667eea; }
    .dataframe { border-radius: 10px; overflow: hidden; }
    </style>
    """, unsafe_allow_html=True)

# ═════════════════════════════════════════════
# CORE CONFIGURATION
# ═════════════════════════════════════════════

def get_llm(api_key: str, temperature: float = 0):
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=temperature
    )

def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 JPEG string for Gemini multimodal input."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def extract_text_from_response(response):
    """Extract plain text from agent response dict."""
    if isinstance(response, dict) and "messages" in response:
        last = response["messages"][-1]
        content = last.content if hasattr(last, 'content') else last.get('content', '')
        if isinstance(content, list):
            return '\n'.join(
                item.get('text','') if isinstance(item, dict) else str(item)
                for item in content if isinstance(item, dict) and item.get('type')=='text' or isinstance(item, str)
            )
        return str(content)
    return str(response)

# ═════════════════════════════════════════════
# YOUTUBE RAG UTILITIES
# ═════════════════════════════════════════════

def extract_youtube_id(url: str) -> str | None:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
        r"(?:embed\/)([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_youtube_transcript(video_id: str) -> list | None:
    """
    Fetch transcript from YouTube using youtube-transcript-api.
    Returns list of {text, start, duration} dicts, or None if unavailable.
    """
    if not TRANSCRIPT_AVAILABLE:
        return None
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception:
        try:
            # Try auto-generated or any available language
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = transcript_list.find_generated_transcript(['en']).fetch()
            return transcript
        except Exception:
            return None

def get_youtube_metadata(video_id: str) -> dict:
    """Fetch basic YouTube video metadata via yt-dlp (no download)."""
    if not YTDLP_AVAILABLE:
        return {"title": "Unknown", "description": "", "duration": 0, "uploader": "Unknown"}
    try:
        ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            return {
                "title":       info.get("title", "Unknown"),
                "description": (info.get("description", "") or "")[:2000],
                "duration":    info.get("duration", 0),
                "uploader":    info.get("uploader", "Unknown"),
                "view_count":  info.get("view_count", 0),
                "upload_date": info.get("upload_date", ""),
                "thumbnail":   info.get("thumbnail", ""),
                "chapters":    info.get("chapters", []),
            }
    except Exception as e:
        return {"title": "Unknown", "description": "", "duration": 0, "uploader": "Unknown"}

def format_transcript_as_chunks(transcript: list, chunk_duration: int = 60) -> list:
    """
    Group transcript entries into ~chunk_duration second segments.
    Returns list of strings like "[0s-60s] combined text..."
    """
    if not transcript:
        return []
    chunks = []
    current_text = []
    chunk_start  = 0
    current_end  = 0

    for entry in transcript:
        start = entry["start"]
        text  = entry["text"].strip().replace("\n", " ")
        if start - chunk_start >= chunk_duration and current_text:
            chunks.append(f"[{int(chunk_start)}s-{int(current_end)}s] " + " ".join(current_text))
            current_text = [text]
            chunk_start  = start
        else:
            current_text.append(text)
        current_end = start + entry.get("duration", 0)

    if current_text:
        chunks.append(f"[{int(chunk_start)}s-{int(current_end)}s] " + " ".join(current_text))
    return chunks

def generate_gemini_summary_chunks(llm, metadata: dict) -> list:
    """
    When no transcript is available, ask Gemini to generate content
    based on title + description for indexing.
    """
    prompt = f"""Based on the following YouTube video metadata, generate 10-15 detailed topic segments
that likely appear in this video. Format each as:
[Topic N] <description of what this segment covers>

Title: {metadata.get('title','')}
Uploader: {metadata.get('uploader','')}
Description: {metadata.get('description','')}
Duration: {metadata.get('duration',0)} seconds

Generate informative segment descriptions:"""
    try:
        resp = llm.invoke(prompt)
        lines = [l.strip() for l in resp.content.strip().splitlines() if l.strip()]
        return lines
    except Exception:
        return [f"[Overview] {metadata.get('title', 'Video content')}"]

def build_yt_rag_index(chunks: list) -> FAISS:
    """Build a FAISS vector store from text chunks."""
    from langchain_core.documents import Document
    docs = [Document(page_content=chunk, metadata={"source": "youtube"}) for chunk in chunks]
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

def seconds_to_hms(seconds: float) -> str:
    """Convert seconds to HH:MM:SS or MM:SS string."""
    seconds = int(seconds)
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"

# ═════════════════════════════════════════════
# TOOL DEFINITIONS
# ═════════════════════════════════════════════

def create_eda_tools(df: pd.DataFrame):
    @tool
    def get_data_summary() -> str:
        """Returns comprehensive info about the dataset: columns, types, missing values, statistics."""
        summary = {
            "columns": df.columns.tolist(),
            "types":   df.dtypes.astype(str).to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "shape":   df.shape,
            "numeric_summary": df.describe().to_dict() if len(df.select_dtypes(include='number').columns) > 0 else {}
        }
        return json.dumps(summary, indent=2)

    @tool
    def generate_visualization(plot_type: str, column: str) -> str:
        """Creates histogram or boxplot. plot_type: 'histogram' or 'boxplot'. column: exact column name."""
        fig, ax = plt.subplots(figsize=(10, 6))
        if plot_type.lower() == "histogram":
            sns.histplot(df[column], kde=True, ax=ax, color='#667eea')
            ax.set_title(f'Distribution of {column}', fontsize=14, fontweight='bold')
        elif plot_type.lower() == "boxplot":
            sns.boxplot(x=df[column], ax=ax, color='#764ba2')
            ax.set_title(f'Boxplot of {column}', fontsize=14, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        return "Plot displayed successfully in the UI."

    return [get_data_summary, generate_visualization]

def create_code_generation_tools():
    @tool
    def generate_python_code(description: str) -> str:
        """Generates Python code based on description."""
        return f"Generating code for: {description}"
    @tool
    def explain_code(code_snippet: str) -> str:
        """Explains what a code snippet does."""
        return "Analyzing code snippet..."
    return [generate_python_code, explain_code]

def create_web_search_tools():
    @tool
    def search_information(query: str) -> str:
        """Searches for information on the web (simulated)."""
        return f"🔍 Searching for: {query}\n\nNote: Web search is simulated in this demo."
    return [search_information]

def create_sql_tools():
    @tool
    def generate_sql_query(description: str) -> str:
        """Generates SQL query from natural language."""
        return f"Generating SQL for: {description}"
    @tool
    def explain_sql_query(query: str) -> str:
        """Explains what an SQL query does."""
        return "Explaining SQL query..."
    return [generate_sql_query, explain_sql_query]

# ═════════════════════════════════════════════
# STREAMLIT UI SETUP
# ═════════════════════════════════════════════

st.set_page_config(
    page_title="🤖 Gemini AI Nexus",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🤖"
)
load_custom_css()

# ═════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════

with st.sidebar:
    st.markdown("# 🤖 AI Nexus Control")
    st.markdown("---")

    api_keys = [
        'AIzaSyBxnhFqUTKTNk9_4ku3EHRjykuCHPSGXO4',
        'AIzaSyBzIhFqiVx-uGD3-q9HzglYTBkW6kIy5bo',
        'AIzaSyD4jfyDs-w-mcpw3h45cmXdQMttmb4UoME',
        'AIzaSyBegUDL9QDxBAmFw7qERw4Rf2GkDaNY3YI',
        'AIzaSyDU3B_BxKtqRqlqoZbtDtUYWiGB0BDsKNA',
        'AIzaSyDWZqrmM7-WTK-06ieW9ay6ND5gVG90IpM'
    ]
    if 'api_key' not in st.session_state:
        st.session_state.api_key = random.choice(api_keys)
    api_key = st.session_state.api_key
    st.success("✅ API Key Auto-Selected")

    st.markdown("---")
    st.markdown("### 🎯 Select Agent")
    chat_mode = st.selectbox(
        "Choose your AI assistant",
        [
            "💬 General Chat",
            "📊 Data Analyst",
            "📄 Document RAG",
            "💻 Code Generator",
            "🔍 Web Research",
            "🗄️ SQL Assistant",
            "🎨 Creative Writer",
            "🎬 Video RAG"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    with st.expander("⚙️ Advanced Settings"):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
        st.info("Higher temperature = more creative responses")

    st.markdown("---")
    agent_info = {
        "💬 General Chat":   "General purpose conversational AI assistant",
        "📊 Data Analyst":   "Analyze CSV files with visualizations",
        "📄 Document RAG":   "Question-answering from PDF documents",
        "💻 Code Generator": "Generate and explain code snippets",
        "🔍 Web Research":   "Research assistant with web search",
        "🗄️ SQL Assistant":  "Generate and explain SQL queries",
        "🎨 Creative Writer":"Creative content generation",
        "🎬 Video RAG":      "Upload a video and chat about its content using vision AI"
    }
    st.markdown(f"**Current Agent:**\n{agent_info.get(chat_mode, '')}")

    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    if 'message_count' not in st.session_state:
        st.session_state.message_count = 0
    st.metric("Messages", st.session_state.message_count)

    llm = get_llm(api_key, temperature)

# ═════════════════════════════════════════════
# MAIN HEADER
# ═════════════════════════════════════════════

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("# 🤖 Gemini AI Nexus")
    st.markdown(f"*{chat_mode} - Powered by Google Gemini*")
with col2:
    st.markdown(f"### {datetime.now().strftime('%I:%M %p')}")
    st.markdown(f"{datetime.now().strftime('%B %d, %Y')}")

st.markdown("---")

# ═════════════════════════════════════════════
# AGENT IMPLEMENTATIONS
# ═════════════════════════════════════════════

# ── DATA ANALYST ──────────────────────────────
if chat_mode == "📊 Data Analyst":
    st.markdown("""
    <div style='background:rgba(255,255,255,0.1);border-radius:12px;padding:16px;margin-bottom:16px;'>
        <h4 style='margin:0'>📊 Data Analyst Agent</h4>
        <p style='margin:4px 0 0 0;opacity:0.85;'>Upload a CSV file to explore your data. Use <b>Visualize</b> to generate charts manually, and <b>Chat</b> to ask questions about your dataset.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📁 Upload CSV File", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("📝 Rows",    df.shape[0])
        with col2: st.metric("📊 Columns", df.shape[1])
        with col3: st.metric("🔢 Numeric", len(df.select_dtypes(include='number').columns))
        with col4: st.metric("⚠️ Missing", df.isnull().sum().sum())
        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["👀 Data Preview", "📈 Visualize", "💬 Chat with Data"])

        with tab1:
            st.dataframe(df.head(20), use_container_width=True)
            with st.expander("📋 Column Info"):
                st.dataframe(pd.DataFrame({
                    "Column":  df.columns,
                    "Type":    df.dtypes.astype(str).values,
                    "Missing": df.isnull().sum().values,
                    "Unique":  df.nunique().values
                }), use_container_width=True)

        with tab2:
            st.markdown("#### 📈 Create Visualizations")
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            all_cols     = df.columns.tolist()
            vcol1, vcol2, vcol3 = st.columns([2, 2, 1])
            with vcol1:
                plot_type = st.selectbox("Plot Type", ["Histogram","Boxplot","Scatter Plot","Bar Chart","Line Chart"])
            with vcol2:
                if plot_type == "Scatter Plot":
                    x_col = st.selectbox("X Column", numeric_cols, key="x_col")
                    y_col = st.selectbox("Y Column", numeric_cols, key="y_col")
                elif plot_type in ["Histogram","Boxplot","Line Chart"]:
                    selected_col = st.selectbox("Select Column", numeric_cols)
                else:
                    selected_col = st.selectbox("Select Column", all_cols)
            with vcol3:
                st.markdown("<br>", unsafe_allow_html=True)
                generate_btn = st.button("🎨 Generate", use_container_width=True)
            if generate_btn:
                try:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    if plot_type == "Histogram":
                        sns.histplot(df[selected_col].dropna(), kde=True, ax=ax, color='#667eea')
                        ax.set_title(f'Distribution of {selected_col}', fontweight='bold')
                        ax.set_xlabel(selected_col); ax.set_ylabel("Frequency")
                    elif plot_type == "Boxplot":
                        sns.boxplot(x=df[selected_col].dropna(), ax=ax, color='#764ba2')
                        ax.set_title(f'Boxplot of {selected_col}', fontweight='bold')
                        ax.set_xlabel(selected_col)
                    elif plot_type == "Scatter Plot":
                        pdf = df[[x_col, y_col]].dropna()
                        ax.scatter(pdf[x_col], pdf[y_col], color='#667eea', alpha=0.6, s=50)
                        ax.set_xlabel(x_col); ax.set_ylabel(y_col)
                        ax.set_title(f'{x_col} vs {y_col}', fontweight='bold'); ax.grid(True, alpha=0.3)
                    elif plot_type == "Bar Chart":
                        vc = df[selected_col].value_counts().head(15)
                        ax.bar(vc.index.astype(str), vc.values, color='#667eea')
                        ax.set_title(f'Top values in {selected_col}', fontweight='bold')
                        ax.set_xlabel(selected_col); ax.set_ylabel("Count")
                        plt.xticks(rotation=45, ha='right')
                    elif plot_type == "Line Chart":
                        ax.plot(df[selected_col].dropna().values, color='#667eea', linewidth=1.5)
                        ax.set_title(f'Line Chart of {selected_col}', fontweight='bold')
                        ax.set_xlabel("Index"); ax.set_ylabel(selected_col); ax.grid(True, alpha=0.3)
                    plt.tight_layout(); st.pyplot(fig); plt.close(fig)
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        with tab3:
            msgs = StreamlitChatMessageHistory(key="data_chat_history")
            if len(msgs.messages) == 0:
                msgs.add_ai_message("👋 Dataset loaded! Ask me anything about your data.")
            tools = create_eda_tools(df)
            agent = create_agent(llm, tools)
            for msg in msgs.messages:
                with st.chat_message(msg.type): st.markdown(msg.content)
            if user_input := st.chat_input("Ask about your data..."):
                st.session_state.message_count += 1
                with st.chat_message("human"): st.markdown(user_input)
                with st.chat_message("ai"):
                    with st.spinner("🤔 Analyzing..."):
                        response = agent.invoke({
                            "messages": [
                                {"role": "system", "content": "You are an expert Data Scientist. Use get_data_summary tool. Always use exact column names."},
                                *[{"role": m.type, "content": m.content} for m in msgs.messages],
                                {"role": "user", "content": user_input}
                            ]
                        })
                        output = extract_text_from_response(response)
                        st.markdown(output)
                        msgs.add_user_message(user_input); msgs.add_ai_message(output)

# ── GENERAL CHAT ──────────────────────────────
elif chat_mode == "💬 General Chat":
    st.markdown("""
    <div style='background:rgba(255,255,255,0.1);border-radius:12px;padding:16px;margin-bottom:16px;'>
        <h4 style='margin:0'>💬 General Chat Agent</h4>
        <p style='margin:4px 0 0 0;opacity:0.85;'>A general-purpose AI assistant powered by Gemini. Ask anything — trivia, explanations, advice, summaries, translations, and more.</p>
        <p style='margin:8px 0 0 0;opacity:0.7;'>✅ Multi-turn conversation &nbsp;|&nbsp; ✅ Any topic &nbsp;|&nbsp; ✅ Fast responses</p>
    </div>
    """, unsafe_allow_html=True)
    msgs = StreamlitChatMessageHistory(key="gen_chat_history")
    if len(msgs.messages) == 0:
        msgs.add_ai_message("👋 Hello! I'm your AI assistant. How can I help you today?")
    for msg in msgs.messages:
        with st.chat_message(msg.type): st.markdown(msg.content)
    if prompt := st.chat_input("Type your message..."):
        st.session_state.message_count += 1
        with st.chat_message("human"): st.markdown(prompt)
        with st.chat_message("ai"):
            with st.spinner("💭 Thinking..."):
                res = llm.invoke(prompt)
                st.markdown(res.content)
                msgs.add_user_message(prompt); msgs.add_ai_message(res.content)

# ── VIDEO RAG (YouTube) ───────────────────────
elif chat_mode == "🎬 Video RAG":
    st.markdown("""
    <div style='background:rgba(255,255,255,0.1);border-radius:12px;padding:16px;margin-bottom:16px;'>
        <h4 style='margin:0'>🎬 YouTube Video RAG Agent</h4>
        <p style='margin:4px 0 0 0;opacity:0.85;'>
            Paste any YouTube link and chat with the video's content. The pipeline fetches the transcript
            (or generates content from metadata), indexes it with <b>vector search</b>, and retrieves
            the most relevant moments to answer your questions with precise timestamps.
        </p>
        <p style='margin:8px 0 0 0;opacity:0.7;'>
            ✅ YouTube Transcript &nbsp;|&nbsp; ✅ Timestamped Chunks &nbsp;|&nbsp;
            ✅ Semantic Search &nbsp;|&nbsp; ✅ Gemini-Grounded Answers
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("ℹ️ How does the YouTube RAG pipeline work?"):
        st.markdown("""
        | Step | What happens |
        |------|-------------|
        | 🔗 URL Parsing | Extracts the YouTube video ID from any valid YouTube URL format |
        | 📝 Transcript Fetch | Downloads the official or auto-generated subtitle transcript |
        | 🧩 Chunking | Groups transcript lines into ~60-second segments with timestamps |
        | 🗂️ Vector Indexing | Chunks embedded with MiniLM and stored in FAISS for semantic search |
        | 🔍 Retrieval | Your question retrieves the top-5 most relevant transcript segments |
        | 💬 Grounded Answer | Gemini answers using retrieved context and cites timestamps |

        **Note:** If a video has no transcript, the agent uses the video title and description to generate topic summaries instead.
        """)

    st.markdown("---")

    # ── URL Input ──────────────────────────────
    url_col, btn_col = st.columns([4, 1])
    with url_col:
        yt_url = st.text_input(
            "🔗 YouTube Video URL",
            placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            help="Paste any YouTube URL — full, short (youtu.be), or embed format"
        )
    with btn_col:
        st.markdown("<br>", unsafe_allow_html=True)
        load_btn = st.button("🚀 Load Video", use_container_width=True)

    # ── Process on button click ────────────────
    if load_btn and yt_url:
        video_id = extract_youtube_id(yt_url)
        if not video_id:
            st.error("❌ Could not extract a valid YouTube video ID from that URL. Please check the link.")
        else:
            video_key = f"yt_rag_{video_id}"
            if st.session_state.get("video_rag_key") != video_key:
                st.session_state.video_rag_key      = video_key
                st.session_state.video_id           = video_id
                st.session_state.video_metadata     = None
                st.session_state.video_chunks       = None
                st.session_state.video_vector_store = None
                st.session_state.video_chat_history = []
                st.session_state.video_source_type  = None  # "transcript" or "generated"

            progress = st.progress(0, text="Fetching video metadata...")

            # Step 1: Metadata
            with st.spinner("📡 Fetching video info..."):
                meta = get_youtube_metadata(video_id)
                st.session_state.video_metadata = meta
            progress.progress(25, text="Fetching transcript...")

            # Step 2: Transcript
            with st.spinner("📝 Fetching transcript..."):
                transcript = get_youtube_transcript(video_id)

            if transcript:
                with st.spinner("🧩 Chunking transcript..."):
                    chunks = format_transcript_as_chunks(transcript, chunk_duration=60)
                st.session_state.video_source_type = "transcript"
                st.success(f"✅ Transcript loaded — {len(chunks)} segments indexed.")
            else:
                st.warning("⚠️ No transcript available. Generating topic segments from video metadata...")
                with st.spinner("🤖 Gemini is generating content segments..."):
                    chunks = generate_gemini_summary_chunks(llm, meta)
                st.session_state.video_source_type = "generated"
                st.info(f"ℹ️ Generated {len(chunks)} topic segments from title & description.")

            progress.progress(66, text="Building semantic index...")

            # Step 3: Index
            with st.spinner("🗂️ Building FAISS vector index..."):
                vs = build_yt_rag_index(chunks)
                st.session_state.video_chunks       = chunks
                st.session_state.video_vector_store = vs

            progress.progress(100, text="✅ Ready to chat!")
            st.rerun()

    # ── Show video info + chat after loading ───
    if st.session_state.get("video_id") and st.session_state.get("video_vector_store"):
        meta     = st.session_state.video_metadata or {}
        video_id = st.session_state.video_id

        # Embedded player + metadata
        embed_col, info_col = st.columns([3, 2])
        with embed_col:
            st.markdown(f"""
            <div style='border-radius:12px;overflow:hidden;box-shadow:0 4px 12px rgba(0,0,0,0.3);'>
                <iframe width='100%' height='315'
                    src='https://www.youtube.com/embed/{video_id}'
                    frameborder='0'
                    allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture'
                    allowfullscreen>
                </iframe>
            </div>
            """, unsafe_allow_html=True)

        with info_col:
            st.markdown("#### 📋 Video Info")
            st.markdown(f"**🎬 Title:** {meta.get('title','—')}")
            st.markdown(f"**👤 Channel:** {meta.get('uploader','—')}")
            dur = meta.get('duration', 0)
            st.markdown(f"**⏱️ Duration:** {seconds_to_hms(dur)} ({dur}s)")
            views = meta.get('view_count', 0)
            st.markdown(f"**👁️ Views:** {views:,}" if views else "**👁️ Views:** —")
            date_raw = meta.get('upload_date', '')
            if date_raw and len(date_raw) == 8:
                date_fmt = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:]}"
                st.markdown(f"**📅 Uploaded:** {date_fmt}")
            src_badge = "📝 Real Transcript" if st.session_state.get("video_source_type") == "transcript" else "🤖 AI-Generated Segments"
            st.markdown(f"**📚 Source:** {src_badge}")
            st.markdown(f"**🧩 Chunks Indexed:** {len(st.session_state.video_chunks or [])}")

            # Chapters if available
            chapters = meta.get("chapters", [])
            if chapters:
                with st.expander("📑 Video Chapters"):
                    for ch in chapters:
                        t = seconds_to_hms(ch.get('start_time', 0))
                        st.markdown(f"- `{t}` — {ch.get('title','')}")

        st.markdown("---")

        tab_chat, tab_index = st.tabs(["💬 Chat with Video", "🗂️ Transcript Index"])

        with tab_index:
            st.markdown("#### 🗂️ Indexed Content Segments")
            st.caption("Each segment below is a searchable chunk of the video's transcript or generated content.")
            for i, chunk in enumerate(st.session_state.video_chunks or []):
                with st.expander(f"Segment {i+1}: {chunk[:80]}..."):
                    st.write(chunk)

        with tab_chat:
            st.markdown("#### 💬 Ask Anything About This Video")
            st.caption(f"Chatting about: **{meta.get('title', video_id)}**")

            # Display history
            for chat in st.session_state.video_chat_history:
                with st.chat_message(chat["role"]):
                    st.markdown(chat["content"])

            examples = ["Summarize this video", "What are the main topics covered?", "What happens at the beginning?"]
            if not st.session_state.video_chat_history:
                st.markdown("**💡 Try asking:**")
                ecols = st.columns(3)
                for i, (col, ex) in enumerate(zip(ecols, examples)):
                    with col: st.code(ex, language=None)

            if question := st.chat_input("Ask about the video..."):
                st.session_state.message_count += 1
                st.session_state.video_chat_history.append({"role": "human", "content": question})

                with st.chat_message("human"):
                    st.markdown(question)

                with st.chat_message("ai"):
                    with st.spinner("🔍 Searching transcript..."):
                        retriever     = st.session_state.video_vector_store.as_retriever(search_kwargs={"k": 5})
                        relevant_docs = retriever.invoke(question)
                        context       = "\n\n".join([d.page_content for d in relevant_docs])

                        history_str = ""
                        for prev in st.session_state.video_chat_history[:-1]:
                            role = "User" if prev["role"] == "human" else "Assistant"
                            history_str += f"{role}: {prev['content']}\n"

                        source_note = (
                            "The context below is from the real video transcript with timestamps."
                            if st.session_state.video_source_type == "transcript"
                            else "The context below is AI-generated from the video's title and description (no transcript was available)."
                        )

                        grounded_prompt = f"""You are a helpful YouTube video assistant.
{source_note}
Answer the user's question using the context below. Cite timestamps like [0s-60s] when relevant.
If the answer is not in the context, say so honestly.

Video Title: {meta.get('title', '')}
Channel: {meta.get('uploader', '')}

--- RELEVANT TRANSCRIPT SEGMENTS ---
{context}

--- CONVERSATION HISTORY ---
{history_str}

User question: {question}

Answer clearly and concisely:"""

                        resp = llm.invoke(grounded_prompt)
                        answer = resp.content
                        st.markdown(answer)
                        st.session_state.video_chat_history.append({"role": "ai", "content": answer})

                        with st.expander("🔍 Retrieved Segments Used"):
                            for doc in relevant_docs:
                                st.markdown(f"```\n{doc.page_content}\n```")

        # New video button
        st.markdown("---")
        if st.button("🔄 Load a Different Video", use_container_width=False):
            for key in ["video_rag_key","video_id","video_metadata","video_chunks",
                        "video_vector_store","video_chat_history","video_source_type"]:
                st.session_state.pop(key, None)
            st.rerun()

# ── DOCUMENT RAG ──────────────────────────────
elif chat_mode == "📄 Document RAG":
    st.markdown("""
    <div style='background:rgba(255,255,255,0.1);border-radius:12px;padding:16px;margin-bottom:16px;'>
        <h4 style='margin:0'>📄 Document RAG Agent</h4>
        <p style='margin:4px 0 0 0;opacity:0.85;'>Upload any PDF and ask questions. The agent reads, chunks, and indexes your document using vector search, then answers based on the content.</p>
        <p style='margin:8px 0 0 0;opacity:0.7;'>✅ PDF Q&A &nbsp;|&nbsp; ✅ Semantic Search &nbsp;|&nbsp; ✅ Multi-page Documents</p>
    </div>
    """, unsafe_allow_html=True)
    msgs     = StreamlitChatMessageHistory(key="rag_chat_history")
    pdf_file = st.file_uploader("📎 Upload PDF Document", type="pdf")
    if pdf_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(pdf_file.getvalue()); tmp_path = tmp.name
        with st.spinner("📖 Processing document..."):
            docs    = PyPDFLoader(tmp_path).load()
            splits  = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)
            vs      = FAISS.from_documents(splits, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
            retriever = vs.as_retriever(search_kwargs={"k": 3})
        col1, col2 = st.columns(2)
        with col1: st.metric("📄 Pages", len(docs))
        with col2: st.metric("🧩 Chunks", len(splits))
        st.markdown("---")
        if len(msgs.messages) == 0:
            msgs.add_ai_message("📚 Document loaded! Ask me anything about its content.")
        for msg in msgs.messages:
            with st.chat_message(msg.type): st.markdown(msg.content)
        if query := st.chat_input("Ask a question about the document..."):
            st.session_state.message_count += 1
            with st.chat_message("human"): st.markdown(query)
            msgs.add_user_message(query)
            with st.chat_message("ai"):
                with st.spinner("🔍 Searching..."):
                    ctx  = "\n\n".join([d.page_content for d in retriever.invoke(query)])
                    resp = llm.invoke(f"Use this context to answer. If not found, say so.\n\nContext:\n{ctx}\n\nQuestion: {query}\n\nAnswer:")
                    st.markdown(resp.content); msgs.add_ai_message(resp.content)
        if os.path.exists(tmp_path): os.remove(tmp_path)

# ── CODE GENERATOR ────────────────────────────
elif chat_mode == "💻 Code Generator":
    st.markdown("""
    <div style='background:rgba(255,255,255,0.1);border-radius:12px;padding:16px;margin-bottom:16px;'>
        <h4 style='margin:0'>💻 Code Generator Agent</h4>
        <p style='margin:4px 0 0 0;opacity:0.85;'>Describe what you need and get clean, well-documented code instantly. Python, JavaScript, SQL, Java, C++, and more.</p>
        <p style='margin:8px 0 0 0;opacity:0.7;'>✅ Multi-language &nbsp;|&nbsp; ✅ Explanation &nbsp;|&nbsp; ✅ Debugging &nbsp;|&nbsp; ✅ Best Practices</p>
    </div>
    """, unsafe_allow_html=True)
    msgs  = StreamlitChatMessageHistory(key="code_chat_history")
    tools = create_code_generation_tools()
    agent = create_agent(llm, tools)
    if len(msgs.messages) == 0:
        msgs.add_ai_message("👨‍💻 Hi! I can generate code, explain snippets, and debug. What would you like to code today?")
    for msg in msgs.messages:
        with st.chat_message(msg.type): st.markdown(msg.content)
    if user_input := st.chat_input("Describe what code you need..."):
        st.session_state.message_count += 1
        with st.chat_message("human"): st.markdown(user_input)
        with st.chat_message("ai"):
            with st.spinner("⚡ Generating..."):
                response = agent.invoke({
                    "messages": [
                        {"role": "system", "content": "You are an expert programmer. Generate clean, well-documented code. Use markdown code blocks."},
                        *[{"role": m.type, "content": m.content} for m in msgs.messages],
                        {"role": "user", "content": user_input}
                    ]
                })
                output = extract_text_from_response(response)
                st.markdown(output); msgs.add_user_message(user_input); msgs.add_ai_message(output)

# ── WEB RESEARCH ──────────────────────────────
elif chat_mode == "🔍 Web Research":
    st.markdown("""
    <div style='background:rgba(255,255,255,0.1);border-radius:12px;padding:16px;margin-bottom:16px;'>
        <h4 style='margin:0'>🔍 Web Research Agent</h4>
        <p style='margin:4px 0 0 0;opacity:0.85;'>AI-powered research assistant. Summaries, comparisons, explanations, deep dives on any topic.</p>
        <p style='margin:8px 0 0 0;opacity:0.7;'>✅ Topic Research &nbsp;|&nbsp; ✅ Summarization &nbsp;|&nbsp; ✅ Comparisons &nbsp;|&nbsp; ✅ Fact Finding</p>
    </div>
    """, unsafe_allow_html=True)
    msgs  = StreamlitChatMessageHistory(key="research_chat_history")
    tools = create_web_search_tools()
    agent = create_agent(llm, tools)
    if len(msgs.messages) == 0:
        msgs.add_ai_message("🔬 Hello! I'm your research assistant. What would you like to research?")
    for msg in msgs.messages:
        with st.chat_message(msg.type): st.markdown(msg.content)
    if user_input := st.chat_input("What would you like to research?"):
        st.session_state.message_count += 1
        with st.chat_message("human"): st.markdown(user_input)
        with st.chat_message("ai"):
            with st.spinner("🔍 Researching..."):
                response = agent.invoke({
                    "messages": [
                        {"role": "system", "content": "You are a research assistant. Provide comprehensive, well-sourced answers."},
                        *[{"role": m.type, "content": m.content} for m in msgs.messages],
                        {"role": "user", "content": user_input}
                    ]
                })
                output = extract_text_from_response(response)
                st.markdown(output); msgs.add_user_message(user_input); msgs.add_ai_message(output)

# ── SQL ASSISTANT ─────────────────────────────
elif chat_mode == "🗄️ SQL Assistant":
    st.markdown("""
    <div style='background:rgba(255,255,255,0.1);border-radius:12px;padding:16px;margin-bottom:16px;'>
        <h4 style='margin:0'>🗄️ SQL Assistant Agent</h4>
        <p style='margin:4px 0 0 0;opacity:0.85;'>Convert plain English into SQL. SELECT, JOIN, GROUP BY, CTEs, subqueries. MySQL, PostgreSQL, SQLite, SQL Server.</p>
        <p style='margin:8px 0 0 0;opacity:0.7;'>✅ Query Generation &nbsp;|&nbsp; ✅ Explanation &nbsp;|&nbsp; ✅ Optimization &nbsp;|&nbsp; ✅ Schema Design</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("**💡 Try asking:**")
    ex1, ex2, ex3 = st.columns(3)
    with ex1: st.code("Get top 10 customers by revenue", language=None)
    with ex2: st.code("Join orders with users table",    language=None)
    with ex3: st.code("Monthly sales trend last year",   language=None)
    st.markdown("---")
    msgs  = StreamlitChatMessageHistory(key="sql_chat_history")
    tools = create_sql_tools()
    agent = create_agent(llm, tools)
    if len(msgs.messages) == 0:
        msgs.add_ai_message("🗄️ Hi! I can generate SQL queries, explain them, and optimize performance. What do you need?")
    for msg in msgs.messages:
        with st.chat_message(msg.type): st.markdown(msg.content)
    if user_input := st.chat_input("Describe your SQL query need..."):
        st.session_state.message_count += 1
        with st.chat_message("human"): st.markdown(user_input)
        with st.chat_message("ai"):
            with st.spinner("💾 Processing..."):
                response = agent.invoke({
                    "messages": [
                        {"role": "system", "content": "You are an expert SQL assistant. Format all queries in SQL code blocks."},
                        *[{"role": m.type, "content": m.content} for m in msgs.messages],
                        {"role": "user", "content": user_input}
                    ]
                })
                output = extract_text_from_response(response)
                st.markdown(output); msgs.add_user_message(user_input); msgs.add_ai_message(output)

# ── CREATIVE WRITER ───────────────────────────
elif chat_mode == "🎨 Creative Writer":
    st.markdown("""
    <div style='background:rgba(255,255,255,0.1);border-radius:12px;padding:16px;margin-bottom:16px;'>
        <h4 style='margin:0'>🎨 Creative Writer Agent</h4>
        <p style='margin:4px 0 0 0;opacity:0.85;'>Stories, poems, blog posts, ad copy, lyrics, scripts. Higher creativity mode enabled.</p>
        <p style='margin:8px 0 0 0;opacity:0.7;'>✅ Stories &nbsp;|&nbsp; ✅ Poems &nbsp;|&nbsp; ✅ Blog Posts &nbsp;|&nbsp; ✅ Brainstorming &nbsp;|&nbsp; ✅ Ad Copy</p>
    </div>
    """, unsafe_allow_html=True)
    msgs         = StreamlitChatMessageHistory(key="creative_chat_history")
    creative_llm = get_llm(api_key, temperature=0.7)
    if len(msgs.messages) == 0:
        msgs.add_ai_message("✨ Hello! Stories, poems, articles, ad copy — what shall we create today?")
    for msg in msgs.messages:
        with st.chat_message(msg.type): st.markdown(msg.content)
    if user_input := st.chat_input("What would you like to create?"):
        st.session_state.message_count += 1
        with st.chat_message("human"): st.markdown(user_input)
        with st.chat_message("ai"):
            with st.spinner("✍️ Creating..."):
                response = creative_llm.invoke(user_input)
                st.markdown(response.content)
                msgs.add_user_message(user_input); msgs.add_ai_message(response.content)

# ═════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div style='text-align:center;color:white;padding:20px;'>
    <p>🤖 Powered by Google Gemini AI | Built with Streamlit & LangChain</p>
    <p>Made with ❤️ for AI Enthusiasts</p>
</div>
""", unsafe_allow_html=True)
