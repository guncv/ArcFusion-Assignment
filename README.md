# ArcFusion - Intelligent Multi-Agent RAG System

An advanced multi-agent Retrieval Augmented Generation (RAG) system built with LangGraph, featuring autonomous tool selection, query refinement, and dual-mode evaluation.

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Agent Descriptions](#agent-descriptions)
- [How to Run Locally](#how-to-run-locally)
- [API Documentation](#api-documentation)
- [Future Improvements](#future-improvements)

---

## Architecture Overview

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                           User Request                               │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FastAPI REST API Layer                          │
│                    (POST /api/v1/llm)                                │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      LangGraph Workflow                              │
│                                                                       │
│  ┌──────────────┐           ┌──────────────────┐                   │
│  │InitRouter    │──ambiguous──►Clarification   │                   │
│  │Agent         │           │  Agent           │                   │
│  └──────┬───────┘           └────┬──────┬──────┘                   │
│         │                        │      │                           │
│    clear_question            smalltalk │                           │
│         │                        │   needs_more                     │
│         │                        ▼      ▼                           │
│         │                   ┌────────────────┐                      │
│         │                   │SmallTalk/More  │                      │
│         │                   │Detail Agents   │──► END               │
│         │                   └────────────────┘                      │
│         │                        │                                  │
│         │                   process_query                           │
│         │                        │                                  │
│         ▼                        ▼                                  │
│    ┌─────────────────────────────────┐                             │
│    │    Refined Query Agent          │                             │
│    └───────────────┬─────────────────┘                             │
│                    │                                                │
│                    ▼                                                │
│    ┌─────────────────────────────────┐                             │
│    │       Planner Agent             │                             │
│    │  (Autonomous Tool Selection)    │                             │
│    └───────────────┬─────────────────┘                             │
│                    │                                                │
│              selected_tool                                          │
│                    │                                                │
│         ┌──────────┴──────────┐                                    │
│         │                     │                                    │
│    rag_search           web_search                                 │
│         │                     │                                    │
│         ▼                     ▼                                    │
│    ┌─────────────────────────────────┐                             │
│    │       Tool Executor             │                             │
│    │  (Parallel Query Execution)     │                             │
│    └───────────────┬─────────────────┘                             │
│                    │                                                │
│              search_results                                         │
│                    │                                                │
│                    ▼                                                │
│    ┌─────────────────────────────────┐                             │
│    │    Synthesizer Agent            │                             │
│    │  (Generate Final Response)      │                             │
│    └───────────────┬─────────────────┘                             │
│                    │                                                │
│                    ▼                                                │
│    ┌─────────────────────────────────┐                             │
│    │    Reflection Agent             │                             │
│    │  (Quality Check & Retry Logic)  │                             │
│    └───────────────┬─────────────────┘                             │
│                    │                                                │
│         ┌──────────┴──────────┐                                    │
│    sufficient          insufficient                                │
│         │                     │                                    │
│         ▼                     │                                    │
│       END            (retry loop back to Planner)                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                    (Background Async Process)
                                 │
                                 ▼
                 ┌────────────────────────────────┐
                 │   Evaluation Agent             │
                 │  (Dual-Mode: Production/Full)  │
                 └────────────────────────────────┘
                                 │
                                 ▼
                 ┌────────────────────────────────┐
                 │   PostgreSQL Database          │
                 │  (Evaluation Metrics Storage)  │
                 └────────────────────────────────┘
```

### Data Flow Architecture

```
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│   Vector DB  │       │  Web Search  │       │ Chat History │
│   (Chroma/   │       │   (Tavily)   │       │ (PostgreSQL) │
│  Pinecone)   │       │              │       │              │
└──────┬───────┘       └──────┬───────┘       └──────┬───────┘
       │                      │                       │
       │ retrieval            │ search                │ context
       │                      │                       │
       ▼                      ▼                       ▼
┌─────────────────────────────────────────────────────────┐
│              Workflow State (TypedDict)                  │
│  - user_query, session_id                               │
│  - selected_tool, generated_queries                     │
│  - web_search_results, retrieved_documents              │
│  - response, is_answer_sufficient                       │
│  - orchestration_attempts, orchestration_history        │
└─────────────────────────────────────────────────────────┘
```

### Key Design Patterns

1. **Singleton Pattern**: LLM instances are cached using `@lru_cache` to prevent duplicate instantiation
2. **Factory Pattern**: Vector stores, retrievers, embeddings, and web search providers use factory classes
3. **Decorator Pattern**: `@handle_agent_error` for consistent exception handling across all agents
4. **Observer Pattern**: Background evaluation runs asynchronously without blocking the main workflow
5. **Strategy Pattern**: Dual-mode evaluation (production vs full) with configurable metrics

---

## Agent Descriptions

### 1. **InitRouterAgent** (ReAct Agent)
**Purpose**: Initial query classification to determine if clarification is needed

**Decision Logic**:
- `clear_question`: Query is specific and self-contained → Route to Planner
- `ambiguous`: Query is vague or unclear → Route to Clarification

**Model**: GPT-3.5-turbo (fast, cost-effective)

**Key Features**:
- Does NOT include chat history (first-pass judgment)
- Enhanced prompt with technical question examples
- Reduces unnecessary clarification for clear technical queries

---

### 2. **ClarificationAgent** (ReAct Agent)
**Purpose**: Fine-grained classification of ambiguous queries with chat history context

**Decision Logic**:
- `smalltalk`: Greetings, acknowledgments, social pleasantries → Route to SmallTalk
- `needs_more_detail`: Vague queries without sufficient context → Route to MoreDetail
- `process_query`: Query can be processed with available context → Route to RefinedQuery

**Model**: GPT-4o-mini (better reasoning)

**Key Features**:
- Includes full chat history for context-aware decisions
- Checks if relative references ("more", "it", "that") have sufficient context
- Simplified prompt with clear principles over rigid examples

---

### 3. **SmallTalkAgent** (Runnable Agent)
**Purpose**: Handle casual conversation without information retrieval

**Model**: GPT-3.5-turbo (temperature=0.5 for friendly responses)

**Workflow**: Generates friendly response → END

---

### 4. **MoreDetailAgent** (Runnable Agent)
**Purpose**: Ask clarifying questions when query lacks necessary details

**Model**: GPT-3.5-turbo (temperature=0.5)

**Workflow**: Generates clarifying questions → END

---

### 5. **RefinedQueryAgent** (Runnable Agent)
**Purpose**: Refine and expand ambiguous queries using chat history

**Model**: GPT-4o-mini

**Key Features**:
- Converts chat history to LangChain message format
- Expands vague queries into clear, specific questions
- Updates `user_query` in state with refined version

---

### 6. **PlannerAgent** (Runnable Agent)
**Purpose**: Autonomous tool selection and query generation

**Decision Logic**:
- `rag_search`: Internal documents, company-specific knowledge
- `web_search`: Real-time information, current events, external knowledge
- `none`: No search needed (general knowledge questions)

**Model**: GPT-4o-mini (temperature=0.3)

**Output**:
- `selected_tool`: Tool to use
- `generated_queries`: 1-3 parallel search queries with purposes

**Key Features**:
- Replanning support (uses reflection feedback)
- Tracks orchestration history
- Prevents duplicate queries

---

### 7. **ToolExecutor** (Non-LLM Agent)
**Purpose**: Execute selected tool with parallel query processing

**Workflow**:
1. Receives `selected_tool` and `generated_queries` from state
2. Creates parallel tasks for each query
3. Uses `asyncio.gather()` for concurrent execution
4. Aggregates results with success/failure tracking

**Key Features**:
- Graceful error handling (logs failures but doesn't halt workflow)
- Detailed logging for each worker
- Returns aggregated results to state

**Tools**:
- `RAGRetrievalAgent`: Vector search with semantic chunking
- `WebSearchAgent`: Tavily web search API

---

### 8. **SynthesizerAgent** (Runnable Agent)
**Purpose**: Generate comprehensive final response from search results

**Model**: GPT-3.5-turbo (temperature=0.3, max_tokens=2500)

**Input**:
- User query
- Search results (web or RAG documents)
- Chat history

**Output**:
- `response`: Final answer
- `current_synthesized_response`: Cached for evaluation

**Key Features**:
- Context-aware (uses chat history)
- Cites sources when available
- Balances comprehensiveness with conciseness

---

### 9. **ReflectionAgent** (Runnable Agent)
**Purpose**: Quality check and retry logic

**Model**: GPT-3.5-turbo (temperature=0.0)

**Decision Logic**:
- `is_answer_sufficient=True`: Response is complete → END
- `is_answer_sufficient=False`: Response needs improvement → Loop back to Planner (with feedback)

**Output**:
- `is_answer_sufficient`: Boolean
- `reflection_issues`: Feedback for replanning

**Key Features**:
- Maximum retry attempts (default: 2)
- Provides specific feedback for replanning
- Prevents infinite loops

---

### 10. **EvaluationAgent** (Background Agent)
**Purpose**: Async evaluation and metrics collection (does not block user response)

**Modes**:
- **Production Mode** (default): Fast, quantitative metrics only
  - `retrieval_quality`: Average document scores (RAG)
  - `relevance_score`: Embedding similarity (Web)
  - `confidence_score`: Overall confidence

- **Full Mode**: Comprehensive LLM-based quality checks
  - All production metrics +
  - `faithfulness`: RAG response grounded in context (LLM check)
  - `consistency`: Web response aligned with search results (LLM check)
  - `answer_relevance`: Response addresses query (LLM check)
  - `context_precision`: Retrieved docs are focused (LLM check)

**Model**: GPT-3.5-turbo (temperature=0.0)

**Storage**: PostgreSQL (evaluation metrics table)

**Key Features**:
- Runs in background (non-blocking)
- Configurable mode via `config.evaluation.mode`
- ~80% cost reduction in production mode

---

## How to Run Locally

### Prerequisites

- Docker and Docker Compose installed
- OpenAI API key (required)
- Tavily API key (optional, for web search)
- Pinecone API key (optional, for vector DB)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ArcFusion-Assignment
   ```

2. **Create `.env` file**
   ```bash
   cp .env.example .env
   ```

3. **Configure environment variables** (edit `.env`)
   ```env
   # Required
   OPENAI_API_KEY=sk-...

   # Optional (for web search)
   TAVILY_API_KEY=tvly-...

   # Optional (for Pinecone vector DB)
   PINECONE_API_KEY=...
   PINECONE_ENV=us-east-1-aws

   # Database (default values work with docker-compose)
   POSTGRES_USER=arcfusion
   POSTGRES_PASSWORD=arcfusion_dev_password
   POSTGRES_DB=arcfusion
   POSTGRES_HOST=postgres
   POSTGRES_PORT=5432

   # Vector DB provider (chroma or pinecone)
   VECTOR_DB=chroma
   ```

4. **Build and start services**
   ```bash
   docker-compose up --build
   ```

5. **Wait for initialization**
   - PostgreSQL health check: ~10-15 seconds
   - Auto-ingestion of documents: ~30 seconds (if enabled)
   - Server ready when you see: `Listening at: http://0.0.0.0:8000`

6. **Access the API**
   - API Base URL: http://localhost:8000
   - Interactive Docs: http://localhost:8000/api/v1/docs
   - OpenAPI Spec: http://localhost:8000/api/v1/openapi.json

### Example API Request

```bash
curl -X POST "http://localhost:8000/api/v1/llm/" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the execution accuracy for the Davinci Codex model on the Spider dataset?",
    "session_id": "test-session-123"
  }'
```

### Project Structure

```
ArcFusion-Assignment/
├── docker-compose.yaml          # Multi-container orchestration
├── Dockerfile                   # Python application container
├── gunicorn.conf.py            # Production WSGI server config
├── package.yaml                # Micromamba environment spec
├── config/
│   └── config.dev.yaml         # Application configuration
├── documents/                  # RAG document corpus (auto-ingested)
├── src/
│   ├── main.py                 # FastAPI application entry point
│   ├── api/
│   │   └── routes.py           # API endpoints
│   ├── agent/
│   │   ├── base/               # Base agent classes (ReAct, Runnable)
│   │   ├── clarification/      # Clarification workflow agents
│   │   ├── autonomous/         # Orchestration workflow agents
│   │   └── evaluation/         # Background evaluation agent
│   ├── graph/
│   │   ├── state.py            # Workflow state schema
│   │   └── workflow_graph.py   # LangGraph workflow definition
│   ├── infras/
│   │   ├── llm/                # LLM loader with caching
│   │   ├── embedding/          # Embedding providers (OpenAI)
│   │   ├── vector_db/          # Vector DB providers (Chroma, Pinecone)
│   │   ├── retrieval/          # RAG retrieval strategies
│   │   ├── chunking/           # Document chunking (semantic)
│   │   ├── ingestion/          # Document processing pipeline
│   │   ├── web_search/         # Web search providers (Tavily)
│   │   ├── database/           # PostgreSQL connection
│   │   └── log/                # Centralized logging
│   ├── repositories/
│   │   ├── chat_history.py     # Chat history CRUD
│   │   └── evaluation.py       # Evaluation metrics CRUD
│   ├── prompts/                # LLM prompts (organized by agent)
│   ├── models/                 # SQLAlchemy models
│   ├── services/               # Business logic services
│   └── utils/                  # Utilities and exception handlers
└── README.md
```

### Stopping the Application

```bash
docker-compose down
```

To also remove volumes (database data):
```bash
docker-compose down -v
```

---

## API Documentation

### Main Endpoint

**POST** `/api/v1/llm/`

**Request Body**:
```json
{
  "query": "string",           // Required: User query
  "session_id": "string"       // Required: Session identifier for chat history
}
```

**Response**:
```json
{
  "response": "string",                    // Final generated response
  "session_id": "string",                  // Session identifier
  "selected_tool": "rag_search|web_search|none",
  "generated_queries": [                   // Queries executed by ToolExecutor
    {
      "query": "string",
      "purpose": "string"
    }
  ],
  "orchestration_attempts": 0,             // Number of retry attempts
  "is_answer_sufficient": true,            // Reflection result
  "routing_decision": "clear_question|ambiguous",
  "user_query": "string"                   // Final query (may be refined)
}
```

### Health Check

**GET** `/health`

**Response**:
```json
{
  "status": "healthy"
}
```

---

## Future Improvements

### 1. **Performance Optimizations**

#### Streaming Responses
- Implement Server-Sent Events (SSE) for real-time response streaming
- Allow users to see partial results as they're generated
- Reduce perceived latency for long responses

#### Caching Layer
- Add Redis for caching common queries and intermediate results
- Cache embedding vectors to reduce OpenAI API calls
- Implement semantic cache (match similar queries)

#### Batch Processing
- Batch multiple user requests for embedding generation
- Reduce API call overhead for high-traffic scenarios

### 2. **Advanced RAG Techniques**

#### Hybrid Search
- Combine vector search (semantic) with keyword search (BM25)
- Implement reciprocal rank fusion (RRF) for result merging
- Improve retrieval quality for domain-specific queries

#### Re-ranking
- Add cross-encoder re-ranking after initial retrieval
- Use ColBERT or similar models for better relevance scoring
- Reduce hallucinations by prioritizing most relevant documents

#### Query Decomposition
- Break complex queries into sub-questions
- Execute sub-questions in parallel
- Synthesize results into comprehensive answer

#### Parent-Child Document Retrieval
- Index document chunks but retrieve parent documents
- Provide more context while maintaining retrieval precision

### 3. **Enhanced Evaluation**

#### User Feedback Loop
- Add thumbs up/down buttons for user feedback
- Store feedback in database for continuous improvement
- Train reward models for RLHF (Reinforcement Learning from Human Feedback)

#### A/B Testing Framework
- Test different agent configurations (models, prompts, parameters)
- Collect metrics (latency, cost, satisfaction)
- Automatically select best-performing configurations

#### Automated Quality Metrics
- Track hallucination rate using fact-checking models
- Monitor response relevance with embedding similarity
- Alert on quality degradation

### 4. **Multi-Modal Capabilities**

#### Document Understanding
- Support PDF, DOCX, PPTX with layout preservation
- Extract tables, charts, and images with multimodal LLMs (GPT-4V, Claude 3)
- Enable image-based queries ("What does this diagram show?")

#### Code Understanding
- Add specialized code embedding models (CodeBERT, StarCoder)
- Support code search and explanation
- Integrate with GitHub/GitLab repositories

### 5. **Advanced Orchestration**

#### Dynamic Tool Composition
- Allow multiple tools per workflow (e.g., RAG + Web Search)
- Implement tool chaining (output of one tool feeds into another)
- Add more tools: SQL databases, API calls, calculator, etc.

#### Adaptive Retry Logic
- Implement exponential backoff for failed queries
- Use reinforcement learning to optimize retry strategies
- Add circuit breaker pattern for failing services

#### Multi-Agent Collaboration
- Enable agent-to-agent communication
- Implement consensus mechanisms (multiple agents vote on best answer)
- Add specialized expert agents (code, math, medical, legal)

### 6. **Scalability & Production Readiness**

#### Horizontal Scaling
- Add load balancer (NGINX, HAProxy)
- Deploy multiple worker instances with Kubernetes
- Implement message queue (RabbitMQ, Kafka) for async processing

#### Monitoring & Observability
- Add Prometheus metrics for request rate, latency, error rate
- Implement distributed tracing with OpenTelemetry
- Set up Grafana dashboards for real-time monitoring
- Add alerting for anomalies (PagerDuty, Slack)

#### Security Enhancements
- Add authentication (OAuth 2.0, JWT)
- Implement rate limiting per user/session
- Add input sanitization to prevent prompt injection
- Encrypt sensitive data at rest and in transit

#### Cost Optimization
- Implement token counting and cost tracking per request
- Add user quotas and billing integration
- Use cheaper models (GPT-3.5) for simple tasks, reserve GPT-4 for complex ones
- Cache embeddings and LLM responses to reduce API calls

### 7. **User Experience**

#### Web UI
- Build React/Vue frontend for chat interface
- Add conversation history sidebar
- Show live status updates (thinking, searching, generating)
- Display sources with clickable citations

#### Mobile App
- Develop native iOS/Android apps
- Support offline mode with local vector search
- Push notifications for long-running queries

#### Voice Interface
- Integrate speech-to-text (Whisper)
- Add text-to-speech for responses
- Support conversational AI (wake word detection)

### 8. **Data & Privacy**

#### GDPR Compliance
- Add user consent management
- Implement data deletion workflows (right to be forgotten)
- Provide data export functionality

#### On-Premise Deployment
- Support fully air-gapped deployments
- Use local LLMs (Llama 3, Mistral) instead of OpenAI
- Self-hosted vector databases and search engines

#### Data Anonymization
- Remove PII before storing chat logs
- Use differential privacy for analytics
- Implement data retention policies

---

## Technology Stack

**Core Framework**: LangGraph (for multi-agent orchestration)
**API Framework**: FastAPI
**LLM Providers**: OpenAI (GPT-4o-mini, GPT-3.5-turbo)
**Vector Databases**: Chroma (default), Pinecone
**Embeddings**: OpenAI text-embedding-3-small
**Web Search**: Tavily
**Database**: PostgreSQL (chat history, evaluation metrics)
**Chunking**: LlamaIndex SemanticSplitterNodeParser
**Document Processing**: Unstructured.io
**Logging**: Custom logger with coloredlogs
**Deployment**: Docker, Docker Compose, Gunicorn

---

## License

This project is submitted as part of the ArcFusion assignment.

---

## Contact

For questions or issues, please contact the project maintainer.
