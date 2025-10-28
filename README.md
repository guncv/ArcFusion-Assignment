# 🚀 ArcFusion - Intelligent Multi-Agent Chatbot LLM System

ArcFusion is an advanced multi-agent LLM system built with LangGraph that intelligently routes user queries through specialized agents, performs autonomous retrieval (RAG, Web Search, or Hybrid), and synthesizes high-quality responses with self-assessment and replanning capabilities.

## 🏛️ Architecture Overview

### System Architecture Diagram
![System Architecture](./workflow_diagram.png)

## 🤖 Agent Descriptions

### Clarification Agents
Agents responsible for understanding and refining user queries before processing.

| Agent | Purpose |
|-------|---------|
| **InitRouterAgent** | Initial entry point that classifies user queries and routes to clear_question or ambiguous paths |
| **ClarificationAgent** | Analyzes ambiguous queries and routes to SmallTalk, NeedsMoreDetail, or RefinedQuery |
| **SmallTalkAgent** | Handles casual conversation and greetings with friendly responses |
| **MoreDetailAgent** | Requests additional information from users for vague queries |
| **RefinedQueryAgent** | Transforms clarified queries into optimized search queries for the Planner |

### Autonomous Orchestration Agents
Core agents that perform intelligent retrieval, synthesis, and quality assessment.

| Agent | Purpose |
|-------|---------|
| **PlannerAgent** | Autonomously selects optimal retrieval tool (RAG/Web/Hybrid) and generates search queries |
| **RAGRetrievalAgent** | Retrieves information from internal vector database using semantic search over PDFs |
| **WebSearchAgent** | Retrieves information from external web sources for current events and recent info |
| **HybridRetrievalAgent** | Executes RAG and Web Search in parallel to combine internal and external sources |
| **SynthesizerAgent** | Generates final answers from retrieved information with proper formatting and citations |
| **MetaAssessorAgent** | Evaluates answer quality and triggers replanning if confidence is below threshold (≥0.7) |

### Evaluation Agent
Background agent for comprehensive quality assessment and analytics.

| Agent | Purpose |
|-------|---------|
| **EvaluationAgent** | Performs asynchronous quality evaluation with metrics like faithfulness, relevance, and consistency |

## 🚀 How to Run Locally

### Prerequisites

- Docker and Docker Compose installed
- OpenAI API key (or other LLM provider key)

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/guncv/ArcFusion-Assignment.git
cd ArcFusion-Assignment
```

2. **Create environment file**

Create a `.env` file in the project root with your configuration:

```env
# LLM Configuration
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key
# Application Configuration
ENV=dev

# Database Configuration (matches docker-compose.yaml)
POSTGRES_USER=arcfusion
POSTGRES_PASSWORD=arcfusion_dev_password
POSTGRES_DB=arcfusion
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

# LangSmith for tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://eu.api.smith.langchain.com
LANGCHAIN_API_KEY="your_langsmith_key"
LANGCHAIN_PROJECT=arcfusion-dev
```

3. **Build and run with Docker Compose**

```bash
# Build and start all services
docker compose up --build

# Or use the Makefile
make build
make run
```

The services will start:
- **API Server**: http://localhost:8000
- **PostgreSQL**: localhost:5432

4. **Wait server for ingestion**

![Running](./running.png)

5. **Test the API**

| Endpoint | Method | Request | Response |
|----------|--------|---------|----------|
| **Chat with LLM** | `POST /api/v1/llm/` | ```"user_input": "What did OpenAI release this month?"``` | ```In October 2025, OpenAI made several significant releases and announcements``` |
| **Clear Chat History** | `POST /api/v1/llm/clear-history` | ```-``` | ```"message": "Chat history cleared successfully"``` |
| **Health Check** | `GET /api/v1/llm/health-check` | ```-``` | ```"status": "ok"``` |

### Stopping the Application

```bash
# Stop services
docker compose down

# Or using Makefile
make down

# Clean everything (including volumes)
make clean
```

## 📁 Project Structure

```
ArcFusion-Assignment/
├── src/
│   ├── agent/                      # Agent implementations
│   │   ├── autonomous/             # Core retrieval & orchestration agents
│   │   │   ├── planner_agent.py
│   │   │   ├── rag_retrieval_agent.py
│   │   │   ├── web_search_agent.py
│   │   │   ├── hybrid_retrieval_agent.py
│   │   │   ├── synthesizer_agent.py
│   │   │   └── meta_assessor_agent.py
│   │   ├── clarification/          # Query clarification agents
│   │   │   ├── init_router_agent.py
│   │   │   ├── clarification_agent.py
│   │   │   ├── small_talk_agent.py
│   │   │   ├── more_detail_agent.py
│   │   │   └── refined_query_agent.py
│   │   ├── evaluation/             # Quality evaluation agents
│   │   │   └── evaluation_agent.py
│   │   └── base/                   # Base agent interfaces
│   │       ├── agent_interface.py
│   │       ├── react_agent.py
│   │       └── runnable_agent.py
│   ├── graph/                      # LangGraph workflow
│   │   ├── state.py
│   │   └── workflow_graph.py
│   ├── infras/                     # Infrastructure components
│   │   ├── chunking/               # Document chunking strategies
│   │   ├── database/               # PostgreSQL integration
│   │   ├── embedding/              # Embedding providers (OpenAI)
│   │   ├── ingestion/              # Document ingestion (Unstructured)
│   │   ├── llm/                    # LLM loader & configuration
│   │   ├── retrieval/              # Retrieval strategies (Vector)
│   │   ├── vector_db/              # Vector database (ChromaDB)
│   │   ├── web_search/             # Web search providers (Tavily)
│   │   └── log/                    # LangSmith tracing & logging
│   ├── prompts/                    # Agent-specific prompts
│   │   ├── autonomous/
│   │   ├── clarification/
│   │   └── evaluation/
│   ├── api/                        # FastAPI routes
│   ├── models/                     # Data models
│   ├── repositories/               # Database repositories
│   ├── services/                   # Business logic services
│   ├── config/                     # Configuration management
│   ├── constants/                  # System constants
│   └── main.py                     # Application entry point
├── documents/                      # PDF documents for RAG ingestion
├── config/                         # YAML configuration files
├── docker-compose.yaml             # Docker orchestration
├── Dockerfile                      # Multi-stage build configuration
├── gunicorn.conf.py                # WSGI server configuration
├── Makefile                        # Development commands
└── README.md                       # This file
```

## 🔮 Future Improvements

### 1. Evaluation System Enhancement (High Priority)
- **User Feedback Integration**: Add thumbs up/down rating to capture real quality scores
- **Monitoring Dashboard**: Visualize metrics and trends over time
- **Adaptive Thresholds**: Replace fixed 0.7 threshold with dynamic confidence levels based on performance data
- **Ground Truth Testing**: Build test dataset for benchmarking and regression testing
- **Metric Weight Tuning**: Optimize evaluation metric combinations based on feedback

### 2. RAG System Optimization (High Priority)
- **Replace Ingestion Method**: Currently using LlamaIndex's UnstructuredReader with Java dependencies. Should to replace with a lighter, more efficient alternative like **PyPDF2** or **pypdf** for basic PDF extraction, or use **LangChain's PyPDFLoader** which doesn't require Java
- **Rethink Chunking Strategy**: Current LlamaIndex SemanticSplitterNodeParser is overkill and resource-intensive. Should migrate to simpler chunking:
  - **LangChain RecursiveCharacterTextSplitter** with fixed chunk size (e.g., 500-1000 chars) and overlap
  - **Semantic chunking** alternatives like **NLTK sentence tokenization + similarity-based grouping**
- **Simplify Dependency Chain**: Remove heavy Unstructured.io + LlamaIndex stack. Use direct LangChain document loaders + splitters for better performance and lower costs
- **Find the Right Balance**: Focus on practical efficiency - current setup is too complex for the use case. Simpler = faster = cheaper

### 3. Prompt Optimization (Medium Priority)
- **Reduce Prompt Length**: Current prompts may be verbose. Refactor to be shorter while maintaining clarity and effectiveness
- **Tailor to Agent Use Cases**: Each agent should have prompts optimized specifically for its role - remove generic instructions and focus on what matters
- **Cost Reduction**: Shorter prompts = lower token usage = reduced LLM costs across all 12 agents
- **Implement Prompt Templates**: Create concise, reusable templates that can be version-controlled and A/B tested

### 4. Chat History & Performance (Low Priority)
- **Add Redis Caching**: Move chat history from PostgreSQL-only to hybrid Redis + PostgreSQL for faster retrieval
- **Benefits**: Redis provides sub-millisecond access times for active conversations, PostgreSQL for long-term persistence
- **Implementation**: Use Redis for hot data (recent sessions), PostgreSQL for cold data and analytics
- **Reduce Database Load**: Lighten PostgreSQL workload by serving frequently accessed data from memory

## 📦 Docker Files Included

This repository includes all necessary Docker configuration:

- **Dockerfile**: Multi-stage build using Micromamba for Python environment
  - System dependencies for PDF processing (poppler, tesseract)
  - Java 17 for LlamaIndex document readers
  - Production-ready with Gunicorn

- **docker-compose.yaml**: Complete orchestration
  - API server (FastAPI + Gunicorn)
  - PostgreSQL database with persistent volume
  - Health checks and dependency management
  - Environment variable configuration

- **gunicorn.conf.py**: Production WSGI configuration
  - Worker configuration
  - Logging setup
  - Performance tuning