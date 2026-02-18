import streamlit as st
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import tempfile
from datetime import datetime

import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# LangChain Core & Google
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Modern Agent Import - 2026 Standard
from langchain.agents import create_agent  # <--- This replaces both Executor and ToolCallingAgent
from langchain.tools import tool
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# RAG Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
# ═════════════════════════════════════════════
# CUSTOM CSS FOR ENHANCED UI
# ═════════════════════════════════════════════

def load_custom_css():
    st.markdown("""
    <style>
    /* Main app styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Card-like containers */
    .stApp > header {
        background-color: transparent;
    }

    /* Chat messages */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Headers */
    h1, h2, h3 {
        color: white !important;
        font-weight: 700 !important;
    }

    /* Input boxes */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #667eea;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }

    /* File uploader */
    .stFileUploader {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        color: #667eea;
        font-weight: 700;
    }

    /* Info boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }

    /* Dataframe */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
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

def extract_text_from_response(response):
    """Extract text content from agent response, handling different formats."""
    if isinstance(response, dict) and "messages" in response:
        last_message = response["messages"][-1]

        # Handle AIMessage object
        if hasattr(last_message, 'content'):
            content = last_message.content
            # If content is a list (with tool calls), extract text parts
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                    elif isinstance(item, str):
                        text_parts.append(item)
                return '\n'.join(text_parts) if text_parts else str(content)
            return str(content)

        # Handle dict message
        elif isinstance(last_message, dict):
            content = last_message.get('content', '')
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                    elif isinstance(item, str):
                        text_parts.append(item)
                return '\n'.join(text_parts) if text_parts else str(content)
            return str(content)

        return str(last_message)

    return str(response)

# ═════════════════════════════════════════════
# TOOL DEFINITIONS FOR AGENTS
# ═════════════════════════════════════════════

def create_eda_tools(df: pd.DataFrame):
    @tool
    def get_data_summary() -> str:
        """Returns comprehensive information about the dataset including columns, types, missing values, and basic statistics."""
        summary = {
            "columns": df.columns.tolist(),
            "types": df.dtypes.astype(str).to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "shape": df.shape,
            "numeric_summary": df.describe().to_dict() if len(df.select_dtypes(include='number').columns) > 0 else {}
        }
        return json.dumps(summary, indent=2)

    @tool
    def generate_visualization(plot_type: str, column: str) -> str:
        """
        Creates a plot. plot_type must be 'histogram' or 'boxplot'.
        column must be the exact column name from the dataset.
        Returns 'Success' if the plot is rendered.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        if plot_type.lower() == "histogram":
            sns.histplot(df[column], kde=True, ax=ax, color='#667eea')
            ax.set_title(f'Distribution of {column}', fontsize=14, fontweight='bold')
        elif plot_type.lower() == "boxplot":
            sns.boxplot(x=df[column], ax=ax, color='#764ba2')
            ax.set_title(f'Boxplot of {column}', fontsize=14, fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        return "Plot displayed successfully in the UI."

    return [get_data_summary, generate_visualization]

def create_code_generation_tools():
    @tool
    def generate_python_code(description: str) -> str:
        """Generates Python code based on the description provided."""
        return f"I'll generate code for: {description}"

    @tool
    def explain_code(code_snippet: str) -> str:
        """Explains what a code snippet does."""
        return f"Analyzing code snippet..."

    return [generate_python_code, explain_code]

def create_web_search_tools():
    @tool
    def search_information(query: str) -> str:
        """Searches for information on the web (simulated)."""
        return f"🔍 Searching for: {query}\n\nNote: Web search is simulated in this demo. Integrate with actual search API for real functionality."

    return [search_information]

def create_sql_tools():
    @tool
    def generate_sql_query(description: str) -> str:
        """Generates SQL query based on natural language description."""
        return f"SQL query will be generated for: {description}"

    @tool
    def explain_sql_query(query: str) -> str:
        """Explains what an SQL query does."""
        return f"Explaining SQL query..."

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
# SIDEBAR CONFIGURATION
# ═════════════════════════════════════════════

with st.sidebar:
    st.markdown("# 🤖 Nexus AI")
    st.markdown("---")

    # Auto API Key Selection
    api_keys = [
        'AIzaSyBxnhFqUTKTNk9_4ku3EHRjykuCHPSGXO4',
        'AIzaSyBzIhFqiVx-uGD3-q9HzglYTBkW6kIy5bo',
        'AIzaSyD4jfyDs-w-mcpw3h45cmXdQMttmb4UoME',
        'AIzaSyBegUDL9QDxBAmFw7qERw4Rf2GkDaNY3YI',
        'AIzaSyDU3B_BxKtqRqlqoZbtDtUYWiGB0BDsKNA',
        'AIzaSyDWZqrmM7-WTK-06ieW9ay6ND5gVG90IpM'
    ]

    # Pick a random key once per session and keep it stable
    if 'api_key' not in st.session_state:
        st.session_state.api_key = random.choice(api_keys)

    api_key = st.session_state.api_key
    st.success("✅ API Key Auto-Selected")

    st.markdown("---")

    # Agent Selection
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
            "🖼️ Image Analyzer"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Model Settings
    with st.expander("⚙️ Advanced Settings"):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
        st.info("Higher temperature = more creative responses")

    st.markdown("---")

    # Agent Info
    agent_info = {
        "💬 General Chat": "General purpose conversational AI assistant",
        "📊 Data Analyst": "Analyze CSV files with visualizations",
        "📄 Document RAG": "Question-answering from PDF documents",
        "💻 Code Generator": "Generate and explain code snippets",
        "🔍 Web Research": "Research assistant with web search",
        "🗄️ SQL Assistant": "Generate and explain SQL queries",
        "🎨 Creative Writer": "Creative content generation",
        "🖼️ Image Analyzer": "Analyze images using ViT Transformers"
    }

    st.markdown(f"**Current Agent:**\n{agent_info.get(chat_mode, '')}")

    # Stats
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

# --- DATA ANALYST AGENT ---
if chat_mode == "📊 Data Analyst":

    # Agent Info Banner
    st.markdown("""
    <div style='background: rgba(255,255,255,0.1); border-radius: 12px; padding: 16px; margin-bottom: 16px;'>
        <h4 style='margin:0'>📊 Data Analyst Agent</h4>
        <p style='margin:4px 0 0 0; opacity:0.85;'>Upload a CSV file to explore your data. Use the <b>Visualize</b> tab to generate charts manually, and the <b>Chat</b> tab to ask questions about your dataset.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📁 Upload CSV File", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Dataset Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📝 Rows", df.shape[0])
        with col2:
            st.metric("📊 Columns", df.shape[1])
        with col3:
            st.metric("🔢 Numeric", len(df.select_dtypes(include='number').columns))
        with col4:
            st.metric("⚠️ Missing", df.isnull().sum().sum())

        st.markdown("---")

        # Tabs: Preview | Visualize | Chat
        tab1, tab2, tab3 = st.tabs(["👀 Data Preview", "📈 Visualize", "💬 Chat with Data"])

        # ── TAB 1: Data Preview ──
        with tab1:
            st.dataframe(df.head(20), use_container_width=True)
            with st.expander("📋 Column Info"):
                info_df = pd.DataFrame({
                    "Column": df.columns,
                    "Type": df.dtypes.astype(str).values,
                    "Missing": df.isnull().sum().values,
                    "Unique": df.nunique().values
                })
                st.dataframe(info_df, use_container_width=True)

        # ── TAB 2: Manual Visualization ──
        with tab2:
            st.markdown("#### 📈 Create Visualizations")

            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            all_cols = df.columns.tolist()

            vcol1, vcol2, vcol3 = st.columns([2, 2, 1])

            with vcol1:
                plot_type = st.selectbox("Plot Type", [
                    "Histogram", "Boxplot", "Scatter Plot", "Bar Chart", "Line Chart"
                ])

            with vcol2:
                if plot_type == "Scatter Plot":
                    x_col = st.selectbox("X Column", numeric_cols, key="x_col")
                    y_col = st.selectbox("Y Column", numeric_cols, key="y_col")
                elif plot_type in ["Histogram", "Boxplot", "Line Chart"]:
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
                        ax.set_xlabel(selected_col)
                        ax.set_ylabel("Frequency")

                    elif plot_type == "Boxplot":
                        sns.boxplot(x=df[selected_col].dropna(), ax=ax, color='#764ba2')
                        ax.set_title(f'Boxplot of {selected_col}', fontweight='bold')
                        ax.set_xlabel(selected_col)

                    elif plot_type == "Scatter Plot":
                        plot_df = df[[x_col, y_col]].dropna()
                        ax.scatter(plot_df[x_col], plot_df[y_col], color='#667eea', alpha=0.6, s=50)
                        ax.set_xlabel(x_col)
                        ax.set_ylabel(y_col)
                        ax.set_title(f'{x_col} vs {y_col}', fontweight='bold')
                        ax.grid(True, alpha=0.3)

                    elif plot_type == "Bar Chart":
                        value_counts = df[selected_col].value_counts().head(15)
                        ax.bar(value_counts.index.astype(str), value_counts.values, color='#667eea')
                        ax.set_title(f'Top values in {selected_col}', fontweight='bold')
                        ax.set_xlabel(selected_col)
                        ax.set_ylabel("Count")
                        plt.xticks(rotation=45, ha='right')

                    elif plot_type == "Line Chart":
                        ax.plot(df[selected_col].dropna().values, color='#667eea', linewidth=1.5)
                        ax.set_title(f'Line Chart of {selected_col}', fontweight='bold')
                        ax.set_xlabel("Index")
                        ax.set_ylabel(selected_col)
                        ax.grid(True, alpha=0.3)

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                except Exception as e:
                    st.error(f"Error generating plot: {str(e)}")

        # ── TAB 3: Chat ──
        with tab3:
            msgs = StreamlitChatMessageHistory(key="data_chat_history")
            if len(msgs.messages) == 0:
                msgs.add_ai_message("👋 Dataset loaded! Ask me anything about your data — statistics, patterns, insights, or specific column details.")

            tools = create_eda_tools(df)
            agent = create_agent(llm, tools)

            for msg in msgs.messages:
                with st.chat_message(msg.type):
                    st.markdown(msg.content)

            if user_input := st.chat_input("Ask about your data..."):
                st.session_state.message_count += 1
                with st.chat_message("human"):
                    st.markdown(user_input)
                with st.chat_message("ai"):
                    with st.spinner("🤔 Analyzing..."):
                        response = agent.invoke({
                            "messages": [
                                {"role": "system", "content": """You are an expert Data Scientist.
You have access to get_data_summary tool. Use it to answer questions about the dataset.
Always use exact column names. Provide clear insights and statistics."""},
                                *[{"role": m.type, "content": m.content} for m in msgs.messages],
                                {"role": "user", "content": user_input}
                            ]
                        })
                        output = extract_text_from_response(response)
                        st.markdown(output)
                        msgs.add_user_message(user_input)
                        msgs.add_ai_message(output)

# --- GENERAL CHAT AGENT ---
elif chat_mode == "💬 General Chat":
    st.markdown("""
    <div style='background: rgba(255,255,255,0.1); border-radius: 12px; padding: 16px; margin-bottom: 16px;'>
        <h4 style='margin:0'>💬 General Chat Agent</h4>
        <p style='margin:4px 0 0 0; opacity:0.85;'>A general-purpose AI assistant . Ask anything — trivia, explanations, advice, summaries, translations, and more.</p>
        <p style='margin:8px 0 0 0; opacity:0.7;'>✅ Multi-turn conversation &nbsp;|&nbsp; ✅ Any topic &nbsp;|&nbsp; ✅ Fast responses</p>
    </div>
    """, unsafe_allow_html=True)

    msgs = StreamlitChatMessageHistory(key="gen_chat_history")

    if len(msgs.messages) == 0:
        msgs.add_ai_message("👋 Hello Aastik! I'm your AI assistant. How can I help you today?")

    for msg in msgs.messages:
        with st.chat_message(msg.type):
            st.markdown(msg.content)

    if prompt := st.chat_input("Type your message..."):
        st.session_state.message_count += 1
        with st.chat_message("human"):
            st.markdown(prompt)
        with st.chat_message("ai"):
            with st.spinner("💭 Thinking..."):
                res = llm.invoke(prompt)
                st.markdown(res.content)
                msgs.add_user_message(prompt)
                msgs.add_ai_message(res.content)

# --- IMAGE ANALYZER AGENT ---
elif chat_mode == "🖼️ Image Analyzer":
    st.markdown("""
    <div style='background: rgba(255,255,255,0.1); border-radius: 12px; padding: 16px; margin-bottom: 16px;'>
        <h4 style='margin:0'>🖼️ Image Analyzer Agent</h4>
        <p style='margin:4px 0 0 0; opacity:0.85;'>Upload any image and get a deep analysis powered by <b>Google ViT (Vision Transformer)</b>. The model classifies objects, detects features, and Gemini then provides a rich natural language explanation.</p>
        <p style='margin:8px 0 0 0; opacity:0.7;'>✅ ViT Image Classification &nbsp;|&nbsp; ✅ Top-5 Predictions &nbsp;|&nbsp; ✅ Gemini Explanation &nbsp;|&nbsp; ✅ Confidence Scores</p>
    </div>
    """, unsafe_allow_html=True)

    # Info expander about ViT
    with st.expander("ℹ️ How does ViT work?"):
        st.markdown("""
        **Vision Transformer (ViT)** — originally introduced by Google Brain — applies the Transformer architecture (famous from NLP) directly to images.

        **How it works:**
        1. 🖼️ **Patch Splitting** — The image is divided into fixed-size patches (e.g., 16×16 pixels)
        2. 🔢 **Linear Embedding** — Each patch is flattened and projected into a vector
        3. 📍 **Positional Encoding** — Position info is added so the model knows patch order
        4. 🔄 **Transformer Encoder** — Multi-head self-attention layers learn global relationships between patches
        5. 🏷️ **Classification Head** — A special `[CLS]` token aggregates info and outputs class probabilities

        **Model used:** `google/vit-base-patch16-224` — trained on ImageNet-21k, fine-tuned on ImageNet-1k (1000 classes)
        """)

    st.markdown("---")

    uploaded_img = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png", "webp", "bmp"])

    if uploaded_img:
        from PIL import Image
        import io

        # Display image
        img = Image.open(uploaded_img).convert("RGB")
        col_img, col_info = st.columns([1, 1])

        with col_img:
            st.image(img, caption="Uploaded Image", use_container_width=True)

        with col_info:
            st.markdown("#### 📐 Image Details")
            st.markdown(f"- **Format:** {uploaded_img.type}")
            st.markdown(f"- **Size:** {img.size[0]} × {img.size[1]} px")
            st.markdown(f"- **Mode:** {img.mode}")
            total_pixels = img.size[0] * img.size[1]
            st.markdown(f"- **Total Pixels:** {total_pixels:,}")

        st.markdown("---")

        with st.spinner("🔍 Running ViT classification..."):
            try:
                from transformers import ViTForImageClassification, ViTImageProcessor
                import torch

                # Load ViT model & processor
                model_name = "google/vit-base-patch16-224"
                processor = ViTImageProcessor.from_pretrained(model_name)
                model = ViTForImageClassification.from_pretrained(model_name)
                model.eval()

                # Preprocess image
                inputs = processor(images=img, return_tensors="pt")

                # Run inference
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits

                # Get top-5 predictions
                probs = torch.nn.functional.softmax(logits, dim=-1)[0]
                top5 = torch.topk(probs, 5)
                top5_indices = top5.indices.tolist()
                top5_scores = top5.values.tolist()
                top5_labels = [model.config.id2label[i] for i in top5_indices]

                # Show predictions
                st.markdown("### 🏷️ ViT Top-5 Predictions")
                for i, (label, score) in enumerate(zip(top5_labels, top5_scores)):
                    rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i]
                    col_rank, col_bar = st.columns([1, 3])
                    with col_rank:
                        st.markdown(f"{rank_emoji} **{label.replace('_', ' ').title()}**")
                        st.markdown(f"`{score*100:.2f}%`")
                    with col_bar:
                        st.progress(float(score))

                st.markdown("---")

                # Build labels string for Gemini
                predictions_text = "\n".join(
                    [f"{i+1}. {label.replace('_', ' ')} ({score*100:.2f}%)"
                     for i, (label, score) in enumerate(zip(top5_labels, top5_scores))]
                )

                # Gemini explanation
                st.markdown("### 🤖 Gemini AI Explanation")
                with st.spinner("💭 Generating detailed explanation..."):
                    gemini_prompt = f"""A Vision Transformer (ViT) model analyzed an image and returned these top predictions:

{predictions_text}

Based on these predictions:
1. What is most likely shown in this image?
2. Explain what the ViT model likely detected (shapes, textures, patterns, colors)
3. Why might the top prediction be correct?
4. What real-world context or setting does this image likely show?
5. Any interesting insights about the classification results?

Provide a clear, engaging, and detailed explanation."""

                    gemini_response = llm.invoke(gemini_prompt)
                    st.markdown(gemini_response.content)

                # Visualization of scores
                st.markdown("---")
                st.markdown("### 📊 Confidence Score Chart")
                fig, ax = plt.subplots(figsize=(10, 4))
                colors = ['#667eea', '#764ba2', '#a855f7', '#c084fc', '#e9d5ff']
                short_labels = [l.replace('_', ' ')[:25] for l in top5_labels]
                bars = ax.barh(short_labels[::-1], [s*100 for s in top5_scores[::-1]], color=colors[::-1])
                ax.set_xlabel("Confidence (%)")
                ax.set_title("ViT Top-5 Prediction Confidence", fontweight='bold')
                for bar, score in zip(bars, top5_scores[::-1]):
                    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                            f'{score*100:.2f}%', va='center', fontweight='bold')
                ax.set_xlim(0, max(top5_scores)*100 + 15)
                ax.grid(True, alpha=0.3, axis='x')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            except ImportError:
                st.error("⚠️ Please install required packages: `pip install transformers torch Pillow`")
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")

# --- DOCUMENT RAG AGENT ---
elif chat_mode == "📄 Document RAG":
    st.markdown("""
    <div style='background: rgba(255,255,255,0.1); border-radius: 12px; padding: 16px; margin-bottom: 16px;'>
        <h4 style='margin:0'>📄 Document RAG Agent</h4>
        <p style='margin:4px 0 0 0; opacity:0.85;'>Upload any PDF and ask questions about it. The agent reads, chunks, and indexes your document using vector search, then answers based on the content.</p>
        <p style='margin:8px 0 0 0; opacity:0.7;'>✅ PDF Question Answering &nbsp;|&nbsp; ✅ Semantic Search &nbsp;|&nbsp; ✅ Multi-page Documents</p>
    </div>
    """, unsafe_allow_html=True)

    msgs = StreamlitChatMessageHistory(key="rag_chat_history")

    pdf_file = st.file_uploader("📎 Upload PDF Document", type="pdf")

    if pdf_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(pdf_file.getvalue())
            tmp_path = tmp.name

        with st.spinner("📖 Processing document..."):
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            splits = text_splitter.split_documents(docs)
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(splits, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        col1, col2 = st.columns(2)
        with col1:
            st.metric("📄 Pages", len(docs))
        with col2:
            st.metric("🧩 Chunks Indexed", len(splits))

        st.markdown("---")

        if len(msgs.messages) == 0:
            msgs.add_ai_message("📚 Document loaded! Ask me anything about its content.")

        for msg in msgs.messages:
            with st.chat_message(msg.type):
                st.markdown(msg.content)

        if query := st.chat_input("Ask a question about the document..."):
            st.session_state.message_count += 1
            with st.chat_message("human"):
                st.markdown(query)
            msgs.add_user_message(query)
            with st.chat_message("ai"):
                with st.spinner("🔍 Searching document..."):
                    context_docs = retriever.invoke(query)
                    context_text = "\n\n".join([d.page_content for d in context_docs])
                    rag_prompt = f"""You are a helpful assistant. Use the following context from a document to answer the user's question.
If the answer isn't in the context, say you don't know based on the document.
Provide detailed and accurate answers.

Context:
{context_text}

Question: {query}

Answer:"""
                    response = llm.invoke(rag_prompt)
                    st.markdown(response.content)
                    msgs.add_ai_message(response.content)

        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- CODE GENERATOR AGENT ---
elif chat_mode == "💻 Code Generator":
    st.markdown("""
    <div style='background: rgba(255,255,255,0.1); border-radius: 12px; padding: 16px; margin-bottom: 16px;'>
        <h4 style='margin:0'>💻 Code Generator Agent</h4>
        <p style='margin:4px 0 0 0; opacity:0.85;'>Describe what you need and get clean, well-documented code instantly. Supports Python, JavaScript, SQL, Java, C++, and more.</p>
        <p style='margin:8px 0 0 0; opacity:0.7;'>✅ Multi-language &nbsp;|&nbsp; ✅ Code Explanation &nbsp;|&nbsp; ✅ Debugging Help &nbsp;|&nbsp; ✅ Best Practices</p>
    </div>
    """, unsafe_allow_html=True)

    msgs = StreamlitChatMessageHistory(key="code_chat_history")
    tools = create_code_generation_tools()
    agent = create_agent(llm, tools)

    if len(msgs.messages) == 0:
        msgs.add_ai_message("👨‍💻 Hi! I can help you with:\n- Generating code in Python, JavaScript, Java, etc.\n- Explaining code snippets\n- Debugging and optimization\n\nWhat would you like to code today?")

    for msg in msgs.messages:
        with st.chat_message(msg.type):
            st.markdown(msg.content)

    if user_input := st.chat_input("Describe what code you need..."):
        st.session_state.message_count += 1
        with st.chat_message("human"):
            st.markdown(user_input)
        with st.chat_message("ai"):
            with st.spinner("⚡ Generating code..."):
                response = agent.invoke({
                    "messages": [
                        {"role": "system", "content": """You are an expert programmer proficient in multiple languages.
Help users generate clean, well-documented code.
Explain code clearly and provide best practices.
Format code with proper syntax highlighting using markdown code blocks."""},
                        *[{"role": m.type, "content": m.content} for m in msgs.messages],
                        {"role": "user", "content": user_input}
                    ]
                })
                output = extract_text_from_response(response)
                st.markdown(output)
                msgs.add_user_message(user_input)
                msgs.add_ai_message(output)

# --- WEB RESEARCH AGENT ---
elif chat_mode == "🔍 Web Research":
    st.markdown("""
    <div style='background: rgba(255,255,255,0.1); border-radius: 12px; padding: 16px; margin-bottom: 16px;'>
        <h4 style='margin:0'>🔍 Web Research Agent</h4>
        <p style='margin:4px 0 0 0; opacity:0.85;'>Your AI-powered research assistant. Ask for summaries, comparisons, explanations, or deep dives on any topic. Best used for knowledge synthesis and research tasks.</p>
        <p style='margin:8px 0 0 0; opacity:0.7;'>✅ Topic Research &nbsp;|&nbsp; ✅ Summarization &nbsp;|&nbsp; ✅ Comparisons &nbsp;|&nbsp; ✅ Fact Finding</p>
    </div>
    """, unsafe_allow_html=True)

    msgs = StreamlitChatMessageHistory(key="research_chat_history")
    tools = create_web_search_tools()
    agent = create_agent(llm, tools)

    if len(msgs.messages) == 0:
        msgs.add_ai_message("🔬 Hello! I'm your research assistant. I can help you:\n- Find information on various topics\n- Summarize research findings\n- Compare different sources\n\nWhat would you like to research?")

    for msg in msgs.messages:
        with st.chat_message(msg.type):
            st.markdown(msg.content)

    if user_input := st.chat_input("What would you like to research?"):
        st.session_state.message_count += 1
        with st.chat_message("human"):
            st.markdown(user_input)
        with st.chat_message("ai"):
            with st.spinner("🔍 Researching..."):
                response = agent.invoke({
                    "messages": [
                        {"role": "system", "content": """You are a research assistant that helps users find and synthesize information.
Provide comprehensive, well-sourced answers.
When you don't have current information, acknowledge it."""},
                        *[{"role": m.type, "content": m.content} for m in msgs.messages],
                        {"role": "user", "content": user_input}
                    ]
                })
                output = extract_text_from_response(response)
                st.markdown(output)
                msgs.add_user_message(user_input)
                msgs.add_ai_message(output)

# --- SQL ASSISTANT AGENT ---
elif chat_mode == "🗄️ SQL Assistant":
    st.markdown("""
    <div style='background: rgba(255,255,255,0.1); border-radius: 12px; padding: 16px; margin-bottom: 16px;'>
        <h4 style='margin:0'>🗄️ SQL Assistant Agent</h4>
        <p style='margin:4px 0 0 0; opacity:0.85;'>Convert plain English into SQL queries instantly. Supports SELECT, JOIN, GROUP BY, subqueries, CTEs, and more. Works with MySQL, PostgreSQL, SQLite, and SQL Server syntax.</p>
        <p style='margin:8px 0 0 0; opacity:0.7;'>✅ Query Generation &nbsp;|&nbsp; ✅ Query Explanation &nbsp;|&nbsp; ✅ Optimization Tips &nbsp;|&nbsp; ✅ Schema Design</p>
    </div>
    """, unsafe_allow_html=True)

    # Quick example chips
    st.markdown("**💡 Try asking:**")
    ex1, ex2, ex3 = st.columns(3)
    with ex1:
        st.code("Get top 10 customers by revenue", language=None)
    with ex2:
        st.code("Join orders with users table", language=None)
    with ex3:
        st.code("Monthly sales trend last year", language=None)

    st.markdown("---")

    msgs = StreamlitChatMessageHistory(key="sql_chat_history")
    tools = create_sql_tools()
    agent = create_agent(llm, tools)

    if len(msgs.messages) == 0:
        msgs.add_ai_message("🗄️ Hi! I'm your SQL assistant. I can help you:\n- Generate SQL queries from descriptions\n- Explain complex queries\n- Optimize query performance\n\nWhat SQL task can I help with?")

    for msg in msgs.messages:
        with st.chat_message(msg.type):
            st.markdown(msg.content)

    if user_input := st.chat_input("Describe your SQL query need..."):
        st.session_state.message_count += 1
        with st.chat_message("human"):
            st.markdown(user_input)
        with st.chat_message("ai"):
            with st.spinner("💾 Processing..."):
                response = agent.invoke({
                    "messages": [
                        {"role": "system", "content": """You are an expert SQL database assistant.
Help users write efficient SQL queries and explain database concepts.
Provide queries with proper formatting and best practices.
Format SQL queries in code blocks."""},
                        *[{"role": m.type, "content": m.content} for m in msgs.messages],
                        {"role": "user", "content": user_input}
                    ]
                })
                output = extract_text_from_response(response)
                st.markdown(output)
                msgs.add_user_message(user_input)
                msgs.add_ai_message(output)

# --- CREATIVE WRITER AGENT ---
elif chat_mode == "🎨 Creative Writer":
    st.markdown("""
    <div style='background: rgba(255,255,255,0.1); border-radius: 12px; padding: 16px; margin-bottom: 16px;'>
        <h4 style='margin:0'>🎨 Creative Writer Agent</h4>
        <p style='margin:4px 0 0 0; opacity:0.85;'>Unleash your creativity with AI. Generate stories, poems, blog posts, ad copy, song lyrics, scripts, and more. Higher creativity mode is enabled for imaginative outputs.</p>
        <p style='margin:8px 0 0 0; opacity:0.7;'>✅ Stories & Narratives &nbsp;|&nbsp; ✅ Poems & Lyrics &nbsp;|&nbsp; ✅ Blog Posts &nbsp;|&nbsp; ✅ Brainstorming &nbsp;|&nbsp; ✅ Ad Copy</p>
    </div>
    """, unsafe_allow_html=True)

    msgs = StreamlitChatMessageHistory(key="creative_chat_history")
    creative_llm = get_llm(api_key, temperature=0.7)

    if len(msgs.messages) == 0:
        msgs.add_ai_message("✨ Hello creative soul! I can help you with:\n- Writing stories and narratives\n- Crafting poems and lyrics\n- Creating blog posts and articles\n- Brainstorming creative ideas\n\nWhat shall we create today?")

    for msg in msgs.messages:
        with st.chat_message(msg.type):
            st.markdown(msg.content)

    if user_input := st.chat_input("What would you like to create?"):
        st.session_state.message_count += 1
        with st.chat_message("human"):
            st.markdown(user_input)
        with st.chat_message("ai"):
            with st.spinner("✍️ Creating..."):
                response = creative_llm.invoke(user_input)
                st.markdown(response.content)
                msgs.add_user_message(user_input)
                msgs.add_ai_message(response.content)

# ═════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 20px;'>
    <p>🤖 Powered by Google Gemini AI | Built with Streamlit & LangChain</p>
    <p>Made with ❤️ for AI Enthusiasts</p>
</div>
""", unsafe_allow_html=True)
