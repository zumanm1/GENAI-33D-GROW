# 🤖 AI Network Engineering Assistant

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive **AI-powered network engineering platform** that combines advanced language models, document management, and network automation capabilities. Built with modern UI/UX design and enterprise-grade features.

## 🌟 Features

### 🤖 **AI-Powered Chat Interface**
- **Multiple AI Modes**: LLM-ONLY, RAG (Retrieval-Augmented Generation), and AGENTIC RAG
- **Context-Aware Responses**: Enhanced with uploaded documents and network knowledge
- **Multi-step Reasoning**: Advanced agentic planning for complex network tasks
- **Real-time Processing**: Instant responses with loading indicators

### 📄 **Document Management System**
- **Multi-format Support**: TXT, PDF, JPEG with automatic text extraction
- **Vector Database**: ChromaDB integration for semantic search
- **Progress Tracking**: Real-time upload progress with status indicators
- **Document Analytics**: View and manage ingested documents

### 🌐 **Network Operations**
- **Device Management**: Connect to multiple network devices
- **Command Execution**: Run network commands remotely
- **Configuration Push**: Deploy configurations to devices
- **Real-time Monitoring**: Live status and connection health

### 🧠 **GenAI Network Automation**
- **NLP-to-Cisco CLI**: Natural language to Cisco commands conversion
- **AI Agent Validation**: CrewAI-powered command validation and cleanup
- **Smart Deployment**: Automated configuration deployment with validation
- **Retrieval Operations**: Natural language queries for device information
- **Multi-Mode Support**: Push Configuration, Retrieve Information, Validate Configuration
- **Real-time Validation**: Post-deployment verification with AI-generated show commands
- **Device Inventory**: SQLite-based device management with sample devices

### 📊 **Analytics & Monitoring**
- **Interaction Statistics**: Track AI usage patterns
- **System Health**: Monitor Ollama, ChromaDB, and database status
- **Log Analysis**: Search and analyze system logs
- **User Preferences**: Customizable settings and configurations

### 🎨 **Modern UI/UX**
- **Responsive Design**: Beautiful gradient-based interface
- **Card-based Layout**: Clean, organized information display
- **Real-time Status**: System health indicators
- **Interactive Elements**: Hover effects and smooth transitions

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Ollama Server** (for AI models)
3. **Network Devices** (optional, for network operations)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd GENAI-33-D-GROW
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Ollama server**
   ```bash
   ollama serve
   ```

4. **Pull required models**
   ```bash
   ollama pull llama2:latest
   ```

5. **Initialize the database (for GenAI Network Automation)**
   ```bash
   python init_database.py
   ```

6. **Run the application**
   ```bash
   streamlit run app.py
   ```

7. **Access the app**
   - Open your browser and go to `http://localhost:8501`
   - Register a new user account
   - Start exploring the features!

## 📁 Project Structure

```
GENAI-33-D-GROW/
├── app.py                 # Main Streamlit application
├── genai_network_automation.py  # GenAI Network Automation module
├── init_database.py       # Database initialization script
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .gitignore            # Git ignore rules
├── ai.db                 # AI interaction database (auto-generated)
├── user.db               # User management database (auto-generated)
├── devices.db            # Network devices database (auto-generated)
└── chroma_db/           # Vector database for documents (auto-generated)
```

## 🎯 Usage Guide

### 1. **Dashboard** 🏠
- **Overview**: System statistics and quick actions
- **Metrics**: Total interactions, documents, devices, and models
- **Quick Actions**: Direct access to main features

### 2. **Configuration** ⚙️
- **AI Model Settings**: Choose Ollama models and agent types
- **System Status**: Monitor Ollama, ChromaDB, and database health
- **Embedding Models**: Configure document processing

### 3. **Chat Interface** 💬
- **Mode Selection**: Choose between LLM-ONLY, RAG, or AGENTIC RAG
- **Real-time Chat**: Interactive conversation with AI
- **Context Awareness**: Responses enhanced with uploaded documents

### 4. **Document Management** 📄
- **Upload Documents**: Drag-and-drop file upload
- **Progress Tracking**: Real-time upload progress
- **Document Analytics**: View ingested documents

### 5. **Network Operations** 🌐
- **Device Management**: View and manage network devices
- **Command Execution**: Run network commands remotely
- **Configuration Push**: Deploy device configurations

### 6. **GenAI Network Automation** 🧠
- **Push Configuration**: Convert natural language to Cisco CLI and deploy
- **Retrieve Information**: Ask questions in natural language to get device info
- **Validate Configuration**: Generate and execute validation commands
- **AI Agent Validation**: Automatic command cleanup and validation
- **Device Selection**: Choose from configured network devices

### 7. **Analytics** 📊
- **Interaction Stats**: View AI usage patterns
- **Log Analysis**: Search system logs
- **User Preferences**: Manage personal settings

## 🔧 Configuration

### Network Devices
Configure your network devices in `app.py`:

```python
devices = [
    {"name": "Router1", "ip": "192.168.1.1", "port": 22, "device_type": "cisco_ios"},
    {"name": "Switch1", "ip": "192.168.1.2", "port": 22, "device_type": "cisco_ios"},
    # Add your devices here
]
```

### AI Models
Available Ollama models:
- `llama2:latest` (default)
- `phi3:mini`
- `llama3.2:1b`
- `starling-lm:7b-alpha-q5_K_M`
- And more...

### Agent Types
- **Default RAG Agent**: Basic document-enhanced responses
- **Advanced Agentic Planner**: Multi-step reasoning and planning
- **Network Diagnostic Agent**: Specialized network troubleshooting

## 🛠️ Development

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run development server
streamlit run app.py --server.port 8501
```

### Code Structure
- **Main App**: `app.py` - Streamlit application entry point
- **Database**: SQLite for user data and AI interactions
- **Vector DB**: ChromaDB for document embeddings
- **AI Integration**: Ollama API for language models

### Key Dependencies
- **Streamlit**: Web application framework
- **ChromaDB**: Vector database for document storage
- **Sentence Transformers**: Text embedding models
- **Netmiko**: Network device automation
- **PDF Plumber**: PDF text extraction
- **Jinja2**: Template engine
- **CrewAI**: AI agent framework for command validation
- **LangChain**: Language model integration

## 🔒 Security Considerations

### Current Implementation
- **User Authentication**: Basic username/password system
- **Plaintext Passwords**: For demo purposes only
- **Local Storage**: All data stored locally

### Production Recommendations
- **Password Hashing**: Implement bcrypt or similar
- **Environment Variables**: Use `.env` for sensitive data
- **HTTPS**: Enable SSL/TLS encryption
- **Input Validation**: Sanitize all user inputs
- **Rate Limiting**: Implement API rate limits

## 🐛 Troubleshooting

### Common Issues

#### 1. **Ollama Connection Error**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama server
ollama serve

# Pull required models
ollama pull llama2:latest
```

#### 2. **Model Not Found**
```bash
# List available models
ollama list

# Pull specific model
ollama pull <model-name>
```

#### 3. **Network Device Connection Issues**
- Verify device IP addresses and credentials
- Check network connectivity
- Ensure proper device_type configuration
- Test with simple commands first

#### 4. **Document Upload Issues**
- Check file format (TXT, PDF, JPEG supported)
- Ensure files are not corrupted
- Verify available disk space for ChromaDB

#### 5. **Database Errors**
```bash
# Remove corrupted databases (if needed)
rm ai.db user.db
# Restart the app to recreate databases
```

### Debug Mode
```bash
# Run with debug information
streamlit run app.py --logger.level debug
```

## 📈 Performance Optimization

### Memory Management
- **ChromaDB**: Configure memory limits for large document collections
- **Sentence Transformers**: Use smaller models for faster processing
- **Database**: Regular cleanup of old logs and interactions

### Network Optimization
- **Connection Pooling**: Reuse network connections
- **Timeout Settings**: Configure appropriate timeouts
- **Batch Processing**: Process multiple documents efficiently

## 🤝 Contributing

### Development Workflow
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Code Style
- Follow PEP 8 Python style guide
- Add docstrings to functions
- Include type hints where appropriate
- Write unit tests for new features

### Testing
```bash
# Run tests (when implemented)
python -m pytest tests/

# Run linting
flake8 app.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit** for the amazing web framework
- **Ollama** for local LLM capabilities
- **ChromaDB** for vector database functionality
- **Netmiko** for network automation
- **Sentence Transformers** for text embeddings

## 📞 Support

### Getting Help
- **Documentation**: Check this README first
- **Issues**: Create GitHub issues for bugs
- **Discussions**: Use GitHub discussions for questions
- **Email**: Contact maintainers for urgent issues

### Community
- **GitHub**: [Repository](https://github.com/your-username/GENAI-33-D-GROW)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/GENAI-33-D-GROW/discussions)
- **Issues**: [GitHub Issues](https://github.com/your-username/GENAI-33-D-GROW/issues)

---

**Made with ❤️ for Network Engineers**

*Empowering network professionals with AI-driven automation and intelligent assistance.* 