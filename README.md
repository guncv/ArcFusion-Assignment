# 🚀 ArcFusion - Intelligent Multi-Agent Chatbot LLM System

ArcFusion is an advanced multi-agent LLM system built with LangGraph that intelligently routes user queries through specialized agents, performs autonomous retrieval (RAG, Web Search, or Hybrid), and synthesizes high-quality responses with self-assessment and replanning capabilities.

## 🏛️ Architecture Overview

### System Architecture Diagram
![System Architecture](./workflow_diagram.png)

## 🤖 Agent Descriptions

### Clarification Agents

#### 1. InitRouterAgent
- **Purpose**: Initial entry point that classifies user queries
- **Decisions**: Routes to `clear_question` (directly to Planner) or `ambiguous` (to Clarification)
- **Implementation**: ReAct agent with routing tool

#### 2. ClarificationAgent
- **Purpose**: Analyzes ambiguous queries to determine appropriate handling
- **Decisions**: Routes to SmallTalk, NeedsMoreDetail, or RefinedQuery
- **Implementation**: ReAct agent with classification tools

#### 3. SmallTalkAgent
- **Purpose**: Handles casual conversation and greetings
- **Action**: Provides friendly responses and terminates workflow

#### 4. MoreDetailAgent
- **Purpose**: Requests additional information from users for vague queries
- **Action**: Asks clarifying questions and terminates workflow

#### 5. RefinedQueryAgent
- **Purpose**: Transforms clarified queries into optimized search queries
- **Action**: Enhances query clarity and passes to Planner

### Autonomous Orchestration Agents

#### 6. PlannerAgent
- **Purpose**: Autonomously selects optimal retrieval tool and generates search queries
- **Tools Available**: `rag_search`, `web_search`, `hybrid_search`
- **Output**: Selected tool + optimized query/queries
- **Replanning**: Can be invoked multiple times if Meta-Assessor deems quality insufficient

#### 7. RAGRetrievalAgent
- **Purpose**: Retrieves information from internal vector database (ChromaDB)
- **Strategy**: Semantic search over ingested PDF documents
- **Best For**: Technical questions, paper-specific information

#### 8. WebSearchAgent
- **Purpose**: Retrieves information from external web sources
- **Strategy**: Web search with result extraction
- **Best For**: Current events, biographical information, recent developments

#### 9. HybridRetrievalAgent
- **Purpose**: Executes RAG and Web Search in parallel
- **Strategy**: Combines internal and external information sources
- **Best For**: Complex queries requiring comprehensive context

#### 10. SynthesizerAgent
- **Purpose**: Generates final answers from retrieved information
- **Features**: 
  - Context-aware prompts for RAG, Web, or combined sources
  - Structured response formatting
  - Citation of sources
- **Output**: Comprehensive, well-formatted answer

#### 11. MetaAssessorAgent
- **Purpose**: Evaluates answer quality and triggers replanning if needed
- **Metrics**: 
  - Confidence score (0.0 to 1.0)
  - Quality reasoning
- **Decisions**: 
  - High confidence (≥0.7): END workflow
  - Low confidence (<0.7): Trigger replanning (up to 3 attempts)
- **Prevents**: Hallucinations and low-quality responses

### Evaluation Agents

#### 12. EvaluationAgent
- **Purpose**: Performs comprehensive quality evaluation and analytics (runs asynchronously in background)
- **Execution**: Triggered by Synthesizer after response generation, non-blocking
- **Evaluation Metrics**:
  - **Faithfulness** (RAG): Verifies answer is supported by retrieved documents
  - **Factual Consistency** (Web): Ensures answer aligns with web sources
  - **Answer Relevance**: Confirms answer addresses user's question
  - **Context Precision** (RAG): Assesses if retrieved context is focused
  - **Retrieval Quality**: Quantitative score based on document similarity scores
  - **Semantic Relevance**: Embedding-based similarity between query and results
- **Dynamic Evaluation**: Adapts evaluation strategy based on tool type (RAG/Web/Hybrid)
- **Confidence Calculation**: Weighted combination of all metrics for comprehensive quality score
- **Database Persistence**: Saves evaluation metrics for analytics, monitoring, and continuous improvement
- **Note**: Separate from MetaAssessorAgent; runs in background for analytics while MetaAssessor drives workflow decisions

## 🚀 How to Run Locally

### Prerequisites

- Docker and Docker Compose installed
- At least 4GB of available RAM
- OpenAI API key (or other LLM provider key)

### Setup Steps

1. **Clone the repository**
```bash
git clone <repository-url>
cd ArcFusion-Assignment
```

2. **Create environment file**

Create a `.env` file in the project root with your configuration:

```env
# LLM Configuration
OPENAI_API_KEY=your_openai_api_key_here
# ANTHROPIC_API_KEY=your_anthropic_key  # Optional
# DEEPSEEK_API_KEY=your_deepseek_key     # Optional

# Database Configuration (matches docker-compose.yaml)
POSTGRES_USER=arcfusion
POSTGRES_PASSWORD=arcfusion_dev_password
POSTGRES_DB=arcfusion
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

# Application Configuration
ENV=dev
LOG_LEVEL=INFO

# Optional: LangSmith for tracing
LANGSMITH_API_KEY=your_langsmith_key  # Optional
LANGSMITH_PROJECT=arcfusion           # Optional
LANGSMITH_TRACING=false               # Set to true to enable
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

4. **Place documents for ingestion**

The system auto-ingests PDF documents from the `documents/` folder on startup. The repository includes sample research papers about Text-to-SQL.

5. **Test the API**

Visit the interactive API docs at http://localhost:8000/api/v1/docs or use curl commands.

| Endpoint | Method | Request | Response |
|----------|--------|---------|----------|
| **Chat with LLM** | `POST /api/v1/llm/` | ```bash<br>curl -X POST "http://localhost:8000/api/v1/llm/" \<br>  -H "Content-Type: application/json" \<br>  -d '{<br>    "user_input": "What approaches are used for text-to-SQL?"<br>  }'<br>``` | ```json<br>{<br>  "response": "Based on the research papers, several approaches are used for text-to-SQL conversion:\\n\\n1. **Prompting Techniques**: Zero-shot and few-shot prompting with large language models like GPT-3 and Codex...\\n\\n2. **Deep Learning Models**: Neural network architectures including sequence-to-sequence models with attention mechanisms...\\n\\n3. **Pre-trained Language Models**: Fine-tuning models like BERT, T5, and GPT for SQL generation tasks...\\n\\nThese approaches are discussed in detail in the retrieved documents."<br>}<br>``` |
| **Clear Chat History** | `POST /api/v1/llm/clear-history` | ```bash<br>curl -X POST "http://localhost:8000/api/v1/llm/clear-history" \<br>  -H "Content-Type: application/json"<br>``` | ```json<br>{<br>  "message": "Chat history cleared successfully"<br>}<br>``` |
| **Health Check** | `GET /api/v1/llm/health-check` | ```bash<br>curl -X GET "http://localhost:8000/api/v1/llm/health-check"<br>``` | ```json<br>{<br>  "status": "ok"<br>}<br>``` |

### Available Make Commands

```bash
make build      # Build Docker images
make run        # Start services
make down       # Stop services
make clean      # Remove all containers, images, and volumes
make restart    # Restart services
make ps         # Show running containers
make rebuild    # Clean build and run
```

### Stopping the Application

```bash
# Stop services
docker compose down

# Or using Makefile
make down

# Clean everything (including volumes)
make clean
```

## 🔮 Future Improvements

### 1. Advanced Retrieval Techniques
- **Reranking**: Implement cross-encoder reranking for better retrieval accuracy
- **Query Decomposition**: Break complex queries into sub-questions
- **Contextual Compression**: Use LLM-based compression to extract only relevant passages
- **Hypothetical Document Embeddings (HyDE)**: Generate hypothetical answers and use them for retrieval

### 2. Multi-Modal Support
- **Image Understanding**: Process diagrams, charts, and figures from PDFs
- **Table Extraction**: Enhanced table parsing and querying
- **Audio/Video**: Transcription and search over multimedia content

### 3. Enhanced Memory & Personalization
- **Long-term Memory**: Maintain user preferences and conversation history across sessions
- **User Profiling**: Adapt responses based on user expertise level
- **Conversation Summarization**: Compress long conversation histories for context

### 4. Production-Ready Features
- **Streaming Responses**: Implement SSE (Server-Sent Events) for real-time streaming
- **Rate Limiting**: Protect API endpoints with Redis-based rate limiting
- **Caching Layer**: Cache frequent queries with Redis
- **Authentication & Authorization**: User management with JWT tokens
- **Monitoring Dashboard**: Grafana + Prometheus for system health
- **A/B Testing**: Framework for testing different agent configurations

### 5. Scalability & Performance
- **Horizontal Scaling**: Kubernetes deployment with load balancing
- **Vector DB Optimization**: Migrate to Pinecone/Weaviate for large-scale production
- **Async Batch Processing**: Queue-based processing for heavy workloads (Celery + RabbitMQ)
- **CDN for Documents**: S3 + CloudFront for document storage

### 6. Advanced Agent Capabilities
- **Tool Use**: Allow agents to use external APIs (calculators, database queries, etc.)
- **Multi-Agent Collaboration**: Specialized agents for different domains (finance, legal, technical)
- **Confidence Calibration**: Fine-tune confidence thresholds based on domain
- **Explainability**: Detailed reasoning chains and decision explanations

### 7. Data Quality & Evaluation
- **Automated Testing**: Unit tests for each agent + integration tests
- **Evaluation Metrics**: Track precision, recall, F1 for retrieval quality
- **Human Feedback Loop**: RLHF for continuous improvement
- **Synthetic Data Generation**: Automated test query generation

### 8. Document Management
- **Incremental Indexing**: Update vector DB without full reindexing
- **Document Versioning**: Track document changes over time
- **Source Attribution**: Enhanced citation with page numbers and excerpts
- **Multi-language Support**: Cross-lingual retrieval and generation

### 9. Cost Optimization
- **LLM Caching**: Semantic caching for similar queries
- **Prompt Optimization**: Reduce token usage with compression
- **Local LLM Options**: Support for Llama 3, Mistral for cost-sensitive deployments
- **Smart Routing**: Use cheaper models for simple tasks, premium for complex ones

### 10. User Experience
- **Web UI**: React-based chat interface with rich media support
- **Mobile App**: Native iOS/Android applications
- **Voice Interface**: Speech-to-text and text-to-speech integration
- **Collaborative Features**: Share conversations, annotations, bookmarks

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