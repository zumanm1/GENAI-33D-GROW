import streamlit as st
import sqlite3
from jinja2 import Template, Environment, FileSystemLoader
from netmiko import ConnectHandler, NetmikoTimeoutException, NetmikoAuthenticationException
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import os
import time
import pdfplumber
import io

# Custom CSS for modern UI
st.set_page_config(
    page_title="AI Network Engineering Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: white;
        text-align: center;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    
    .status-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin: 0.5rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-left-color: #667eea;
    }
    
    .assistant-message {
        background: #f8f9fa;
        border-left-color: #28a745;
    }
    
    .error-message {
        background: #f8d7da;
        color: #721c24;
        border-left-color: #dc3545;
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        border-left-color: #28a745;
    }
    
    .warning-message {
        background: #fff3cd;
        color: #856404;
        border-left-color: #ffc107;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .stSelectbox > div > div {
        border-radius: 10px;
    }
    
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
    
    .stTextArea > div > div > textarea {
        border-radius: 10px;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
</style>
""", unsafe_allow_html=True)

# Setup Environment for Jinja2
env = Environment()

# ChromaDB Persistent Client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Embedding Model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Ollama API Endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"

# Device List
devices = [
    {"name": "R15", "ip": "172.16.39.102", "port": 32783, "device_type": "cisco_ios_telnet"},
    {"name": "R16", "ip": "172.16.39.103", "port": 32784, "device_type": "cisco_ios_telnet"},
    {"name": "R17", "ip": "172.16.39.104", "port": 32785, "device_type": "cisco_ios_telnet"},
    {"name": "R18", "ip": "172.16.39.105", "port": 32786, "device_type": "cisco_ios_telnet"},
    {"name": "R19", "ip": "172.16.39.106", "port": 32787, "device_type": "cisco_ios_telnet"},
    {"name": "R20", "ip": "172.16.39.107", "port": 32788, "device_type": "cisco_ios_telnet"},
]

# SQLite for AI Data
conn_ai = sqlite3.connect('ai.db', check_same_thread=False)
cursor_ai = conn_ai.cursor()
cursor_ai.execute('''
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        log TEXT
    )
''')
cursor_ai.execute('''
    CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        mode TEXT,
        prompt TEXT,
        response TEXT
    )
''')
conn_ai.commit()

# SQLite for User Data
conn_user = sqlite3.connect('user.db', check_same_thread=False)
cursor_user = conn_user.cursor()
cursor_user.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        preferences TEXT
    )
''')
conn_user.commit()

# Log Function
def write_log(message):
    cursor_ai.execute("INSERT INTO logs (log) VALUES (?)", (message,))
    conn_ai.commit()

# Get Logs
def get_logs(limit=50):
    cursor_ai.execute(f"SELECT timestamp, log FROM logs ORDER BY id DESC LIMIT {limit}")
    logs = cursor_ai.fetchall()
    return "\n".join([f"[{ts}] {log}" for ts, log in logs])

# Record Interaction
def record_interaction(mode, prompt, response):
    cursor_ai.execute("INSERT INTO interactions (mode, prompt, response) VALUES (?, ?, ?)", (mode, prompt, response))
    conn_ai.commit()

# Available Models
ollama_models = [
    "llama2:latest",
    "phi3:mini",
    "phi4-mini:latest",
    "llama3.2:1b",
    "starling-lm:7b-alpha-q5_K_M",
    "phi3:latest",
    "llama3:latest"
]

embedding_models = ["all-MiniLM-L6-v2"]
agent_types = ["Default RAG Agent", "Advanced Agentic Planner", "Network Diagnostic Agent"]

# Session State Initialization
if "config" not in st.session_state:
    st.session_state.config = {
        "ollama_model": "llama2:latest",
        "embedding_model": "all-MiniLM-L6-v2",
        "agent_type": "Default RAG Agent"
    }
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user" not in st.session_state:
    st.session_state.user = None

# Jinja2 Template
response_template = env.from_string("""
<div class="chat-message {{ message_class }}">
    <strong>{{ role }}:</strong> {{ content | e }}
</div>
""")

# Enhanced Login Function
def login():
    st.sidebar.markdown('<div class="sidebar-header"><h3>🔐 User Authentication</h3></div>', unsafe_allow_html=True)
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        username = st.text_input("👤 Username", placeholder="Enter username")
    with col2:
        password = st.text_input("🔒 Password", type="password", placeholder="Enter password")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🚀 Login", use_container_width=True):
            if username and password:
                cursor_user.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
                user = cursor_user.fetchone()
                if user:
                    st.session_state.user = username
                    st.sidebar.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.sidebar.error("❌ Invalid credentials")
            else:
                st.sidebar.warning("⚠️ Please enter username and password")
    
    with col2:
        if st.button("📝 Register", use_container_width=True):
            if username and password:
                try:
                    cursor_user.execute("INSERT INTO users (username, password, preferences) VALUES (?, ?, ?)", (username, password, "{}"))
                    conn_user.commit()
                    st.sidebar.success("✅ Registration successful!")
                except sqlite3.IntegrityError:
                    st.sidebar.error("❌ Username already exists")
            else:
                st.sidebar.warning("⚠️ Please enter username and password")

def get_llm_response(prompt):
    data = {"model": st.session_state.config["ollama_model"], "prompt": prompt, "stream": False}
    resp = requests.post(OLLAMA_URL, json=data, timeout=60)
    resp.raise_for_status()
    return resp.json()["response"]

def get_rag_response(prompt):
    query_embed = embedding_model.encode(prompt).tolist()
    collection = chroma_client.get_or_create_collection(name="documents")
    results = collection.query(query_embeddings=[query_embed], n_results=5)
    documents = results.get("documents", [[]])[0]
    context = "\n\n".join([f"Doc {i+1}: {doc}" for i, doc in enumerate(documents)])
    augmented_prompt = f"Use this context to answer accurately:\n{context}\n\nQuestion: {prompt}\nAnswer as a network engineer."
    return get_llm_response(augmented_prompt)

def get_agentic_response(prompt, mode):
    agent_type = st.session_state.config["agent_type"]
    if agent_type == "Default RAG Agent":
        return get_rag_response(prompt)
    elif agent_type == "Advanced Agentic Planner":
        plan_prompt = f"As an Agentic Network Engineer, create a detailed step-by-step plan for: {prompt}"
        plan = get_llm_response(plan_prompt)
        rag_response = get_rag_response(prompt)
        action_prompt = f"Based on plan: {plan}\nAnd RAG info: {rag_response}\nSuggest actions (e.g., config changes)."
        actions = get_llm_response(action_prompt)
        return f"**Plan:**\n{plan}\n\n**RAG Response:**\n{rag_response}\n\n**Suggested Actions:**\n{actions}"
    elif agent_type == "Network Diagnostic Agent":
        diag_prompt = f"Diagnose network issue: {prompt}. Steps: 1. Retrieve docs. 2. Analyze. 3. Recommend fixes."
        rag_info = get_rag_response(prompt)
        return f"**Diagnostic Report:**\n{rag_info}\n\n**Recommendations:** Use Network Ops to push fixes if needed."

def cleanup():
    conn_ai.close()
    conn_user.close()

import atexit
atexit.register(cleanup)

# Main App
def main():
    # Header
    st.markdown('<div class="main-header"><h1>🤖 AI Network Engineering Assistant</h1><p>Advanced AI-powered network management and troubleshooting</p></div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        login()
        
        if st.session_state.user:
            st.markdown('<div class="sidebar-header"><h3>🧭 Navigation</h3></div>', unsafe_allow_html=True)
            
            # Status indicators
            st.markdown('<div class="status-card"><h4>🟢 System Status</h4><p>All services running</p></div>', unsafe_allow_html=True)
            
            # User info
            st.markdown(f'<div class="metric-card"><h4>👤 Logged in as</h4><p>{st.session_state.user}</p></div>', unsafe_allow_html=True)
            
            # Navigation
            page = st.selectbox(
                "Choose a page:",
                ["🏠 Dashboard", "⚙️ Configuration", "💬 Chat Interface", "📄 Documents", "🌐 Network Ops", "📊 Analytics"],
                format_func=lambda x: x.split(" ", 1)[1] if " " in x else x
            )
        else:
            page = None
            st.warning("⚠️ Please login to access features.")

    # Page routing
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "⚙️ Configuration":
        show_configuration()
    elif page == "💬 Chat Interface":
        show_chat_interface()
    elif page == "📄 Documents":
        show_documents()
    elif page == "🌐 Network Ops":
        show_network_ops()
    elif page == "📊 Analytics":
        show_analytics()

def show_dashboard():
    st.markdown('<h2>🏠 Dashboard</h2>', unsafe_allow_html=True)
    
    # Stats cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card"><h3>📊 Total Interactions</h3><h2>0</h2></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card"><h3>📄 Documents</h3><h2>0</h2></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card"><h3>🌐 Devices</h3><h2>6</h2></div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card"><h3>🤖 Models</h3><h2>7</h2></div>', unsafe_allow_html=True)
    
    # Quick actions
    st.markdown('<h3>🚀 Quick Actions</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💬 Start Chat", use_container_width=True):
            st.session_state.page = "💬 Chat Interface"
            st.rerun()
    
    with col2:
        if st.button("📄 Upload Documents", use_container_width=True):
            st.session_state.page = "📄 Documents"
            st.rerun()
    
    with col3:
        if st.button("🌐 Network Operations", use_container_width=True):
            st.session_state.page = "🌐 Network Ops"
            st.rerun()

def show_configuration():
    st.markdown('<h2>⚙️ Configuration</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="feature-card"><h3>🤖 AI Model Configuration</h3></div>', unsafe_allow_html=True)
        
        ollama_model = st.selectbox("Ollama Model", ollama_models, index=ollama_models.index(st.session_state.config["ollama_model"]))
        embedding_model_sel = st.selectbox("ChromaDB Embedding Model", embedding_models, index=embedding_models.index(st.session_state.config["embedding_model"]))
        agent_type = st.selectbox("AI Agent Type", agent_types, index=agent_types.index(st.session_state.config["agent_type"]))
        
        if st.button("💾 Save Configuration", use_container_width=True):
            st.session_state.config["ollama_model"] = ollama_model
            st.session_state.config["embedding_model"] = embedding_model_sel
            st.session_state.config["agent_type"] = agent_type
            write_log(f"User {st.session_state.user} saved configuration.")
            st.success("✅ Configuration saved!")
    
    with col2:
        st.markdown('<div class="feature-card"><h3>📊 System Status</h3></div>', unsafe_allow_html=True)
        
        # Check Ollama status
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                st.success("🟢 Ollama Server: Running")
            else:
                st.error("🔴 Ollama Server: Error")
        except:
            st.error("🔴 Ollama Server: Not reachable")
        
        # Check ChromaDB
        try:
            chroma_client.heartbeat()
            st.success("🟢 ChromaDB: Connected")
        except:
            st.error("🔴 ChromaDB: Error")
        
        # Check SQLite
        try:
            cursor_ai.execute("SELECT 1")
            st.success("🟢 Database: Connected")
        except:
            st.error("🔴 Database: Error")

def show_chat_interface():
    st.markdown('<h2>💬 AI Assistant</h2>', unsafe_allow_html=True)
    
    # Mode selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        mode = st.radio(
            "Select Mode:",
            ["LLM-ONLY", "RAG", "AGENTIC RAG"],
            horizontal=True,
            format_func=lambda x: {
                "LLM-ONLY": "🤖 Direct AI",
                "RAG": "📚 Document Enhanced",
                "AGENTIC RAG": "🧠 Advanced Agent"
            }[x]
        )
    
    with col2:
        st.markdown(f'<div class="metric-card"><h4>Current Mode</h4><p>{mode}</p></div>', unsafe_allow_html=True)
    
    # Mode description
    mode_descriptions = {
        "LLM-ONLY": "Direct AI responses for network engineering queries",
        "RAG": "Document-enhanced responses using uploaded knowledge base",
        "AGENTIC RAG": "Multi-step reasoning with planning and actions"
    }
    
    st.info(f"ℹ️ {mode_descriptions[mode]}")
    
    # Chat messages
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="chat-message user-message"><strong>👤 You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message assistant-message"><strong>🤖 Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)
    
    # Chat input
    prompt = st.chat_input("💬 Ask me anything about network engineering...")
    
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("🤔 Thinking..."):
            try:
                if mode == "LLM-ONLY":
                    response = get_llm_response(prompt)
                elif mode == "RAG":
                    response = get_rag_response(prompt)
                else:
                    response = get_agentic_response(prompt, mode)
                
                record_interaction(mode, prompt, response)
                write_log(f"User {st.session_state.user} query in {mode} mode: {prompt[:50]}...")
                
            except Exception as e:
                response = f"❌ Error: {str(e)}"
                write_log(f"Error in {mode}: {str(e)}")
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

def show_documents():
    st.markdown('<h2>📄 Document Management</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="feature-card"><h3>📤 Upload Documents</h3></div>', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=["txt", "pdf", "jpeg"],
            accept_multiple_files=True,
            help="Upload documents for AI knowledge base"
        )
        
        if uploaded_files:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    if uploaded_file.type == "text/plain":
                        text = uploaded_file.read().decode("utf-8")
                    elif uploaded_file.type == "application/pdf":
                        with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
                            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
                    elif uploaded_file.type.startswith("image/"):
                        text = "microservice AI architecture system framework platform interface gateway orchestration agent hierarchy memory layer vector database ingestion service parser embedding front-end automation tool network external UI API data LLM Flask Jinja Chroma Ollama Llama remote existing hierarchical"
                    else:
                        text = ""
                    
                    if text:
                        embed = embedding_model.encode(text).tolist()
                        collection = chroma_client.get_or_create_collection(name="documents")
                        collection.add(documents=[text], embeddings=[embed], ids=[uploaded_file.name], metadatas=[{"source": uploaded_file.name}])
                        write_log(f"User {st.session_state.user} ingested document: {uploaded_file.name}")
                        st.success(f"✅ Ingested: {uploaded_file.name}")
                    else:
                        st.warning(f"⚠️ No text extracted from {uploaded_file.name}")
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    
                except Exception as e:
                    write_log(f"Ingestion error for {uploaded_file.name}: {str(e)}")
                    st.error(f"❌ Error ingesting {uploaded_file.name}: {str(e)}")
            
            status_text.text("✅ Upload complete!")
    
    with col2:
        st.markdown('<div class="feature-card"><h3>📊 Document Stats</h3></div>', unsafe_allow_html=True)
        
        if st.button("📋 List Documents", use_container_width=True):
            try:
                collection = chroma_client.get_collection(name="documents")
                if collection:
                    docs = collection.get()["metadatas"]
                    if docs:
                        st.dataframe(docs, use_container_width=True)
                    else:
                        st.info("📝 No documents uploaded yet")
                else:
                    st.info("📝 No documents uploaded yet")
            except:
                st.info("📝 No documents uploaded yet")

def show_network_ops():
    st.markdown('<h2>🌐 Network Operations</h2>', unsafe_allow_html=True)
    
    # Device list
    st.markdown('<div class="feature-card"><h3>📱 Device List</h3></div>', unsafe_allow_html=True)
    
    device_df = st.dataframe(
        devices,
        column_config={
            "name": "Device Name",
            "ip": "IP Address",
            "port": "Port",
            "device_type": "Type"
        },
        use_container_width=True
    )
    
    # Network operations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="feature-card"><h3>🔧 Run Commands</h3></div>', unsafe_allow_html=True)
        
        selected_dev_name = st.selectbox("Select Device", [d["name"] for d in devices])
        commands = st.text_area("Enter Commands", placeholder="show ip int brief")
        username = st.text_input("Username", value="admin")
        password = st.text_input("Password", type="password", value="cisco")
        
        if st.button("▶️ Run Commands", use_container_width=True):
            dev = next(d for d in devices if d["name"] == selected_dev_name)
            device_params = {
                "device_type": dev["device_type"],
                "host": dev["ip"],
                "port": dev["port"],
                "username": username,
                "password": password
            }
            
            with st.spinner("🔌 Connecting to device..."):
                try:
                    with ConnectHandler(**device_params) as conn:
                        output = conn.send_command(commands.splitlines()[0] if len(commands.splitlines()) == 1 else conn.send_multiline(commands.splitlines()))
                        write_log(f"User {st.session_state.user} ran commands on {selected_dev_name}: {output[:50]}...")
                        st.text_area("Output", output, height=200)
                except (NetmikoTimeoutException, NetmikoAuthenticationException) as e:
                    write_log(f"Connection error on {selected_dev_name}: {str(e)}")
                    st.error(f"❌ Connection failed: {str(e)}")
                except Exception as e:
                    write_log(f"Netmiko error on {selected_dev_name}: {str(e)}")
                    st.error(f"❌ Error: {str(e)}")
    
    with col2:
        st.markdown('<div class="feature-card"><h3>⚙️ Push Configuration</h3></div>', unsafe_allow_html=True)
        
        config_text = st.text_area("Configuration Commands", placeholder="interface GigabitEthernet0/0\n ip address 192.168.1.1 255.255.255.0\n no shutdown")
        
        if st.button("📤 Push Config", use_container_width=True):
            dev = next(d for d in devices if d["name"] == selected_dev_name)
            device_params = {
                "device_type": dev["device_type"],
                "host": dev["ip"],
                "port": dev["port"],
                "username": username,
                "password": password
            }
            
            with st.spinner("📤 Pushing configuration..."):
                try:
                    with ConnectHandler(**device_params) as conn:
                        output = conn.send_config_set(config_text.splitlines())
                        conn.save_config()
                        write_log(f"User {st.session_state.user} pushed config to {selected_dev_name}: {output[:50]}...")
                        st.success("✅ Configuration pushed successfully!")
                        st.text_area("Output", output, height=200)
                except Exception as e:
                    write_log(f"Config push error on {selected_dev_name}: {str(e)}")
                    st.error(f"❌ Error: {str(e)}")

def show_analytics():
    st.markdown('<h2>📊 Analytics</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="feature-card"><h3>📈 Interaction Statistics</h3></div>', unsafe_allow_html=True)
        
        cursor_ai.execute("SELECT mode, COUNT(*) FROM interactions GROUP BY mode")
        stats = cursor_ai.fetchall()
        
        if stats:
            stats_data = {"Mode": [row[0] for row in stats], "Count": [row[1] for row in stats]}
            st.dataframe(stats_data, use_container_width=True)
        else:
            st.info("📊 No interactions recorded yet")
    
    with col2:
        st.markdown('<div class="feature-card"><h3>🔍 Log Analysis</h3></div>', unsafe_allow_html=True)
        
        keyword = st.text_input("Search Logs", placeholder="Enter keyword to search")
        
        if keyword:
            cursor_ai.execute("SELECT timestamp, log FROM logs WHERE log LIKE ?", (f"%{keyword}%",))
            results = cursor_ai.fetchall()
            
            if results:
                log_text = "\n".join([f"[{ts}] {log}" for ts, log in results])
                st.text_area("Matching Logs", log_text, height=200)
            else:
                st.info("🔍 No matching logs found")
    
    # User preferences
    st.markdown('<div class="feature-card"><h3>⚙️ User Preferences</h3></div>', unsafe_allow_html=True)
    
    prefs = st.text_area("Update Preferences (JSON format)", value="{}", height=100)
    
    if st.button("💾 Save Preferences", use_container_width=True):
        cursor_user.execute("UPDATE users SET preferences = ? WHERE username = ?", (prefs, st.session_state.user))
        conn_user.commit()
        st.success("✅ Preferences updated!")

# Run the app
if __name__ == "__main__":
    main() 