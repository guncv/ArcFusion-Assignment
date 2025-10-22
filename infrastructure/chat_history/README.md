# Chat History - Production-Ready Implementation

This module provides a production-ready chat history implementation using LangChain with multiple backend support.

## Features

✅ **Redis Backend** (Production)
- Fast, in-memory storage
- Distributed system support
- Automatic TTL (session expiration)
- Thread-safe operations
- Scales horizontally

✅ **File Backend** (Development/Fallback)
- Local JSON file storage
- Automatic fallback if Redis unavailable
- Good for local development

✅ **Auto-Persistence**
- Messages automatically saved on add
- No manual save() calls needed
- Crash-resistant

## Configuration

Edit `core/config/config.dev.yaml`:

```yaml
chat_history:
  backend: "redis"  # Options: redis, file
  ttl: 3600  # Session expiration (1 hour)

  redis:
    host: "${REDIS_HOST:localhost}"
    port: "${REDIS_PORT:6379}"
    db: "${REDIS_DB:0}"
    password: "${REDIS_PASSWORD:}"
    key_prefix: "chat_history:"

  file:
    persist_dir: "./data/chat_history"
```

## Environment Variables

Add to `.env.dev`:

```bash
REDIS_HOST=redis        # Docker service name or localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=         # Leave empty if no password
```

## Usage

### Basic Usage

```python
from infrastructure.chat_history import get_session_history

# Get chat history for a session
history = get_session_history(session_id="user_123")

# Add messages
history.add_user_message("Hello!")
history.add_ai_message("Hi! How can I help?")

# Get all messages
messages = history.messages

# Get last N messages
recent_messages = history.messages[-5:]  # Last 5 messages
```

### In Agents

```python
from infrastructure.llm.loader import getChatHistory

async def my_agent(state: WorkflowState) -> WorkflowState:
    session_id = state.get("session_id")

    # Get chat history
    chat_history = getChatHistory(session_id)

    # Use it in your prompt
    history_str = "\n".join([
        f"{msg.type}: {msg.content}"
        for msg in chat_history.messages[-5:]  # Last 5 for context
    ])

    # ... process with LLM
```

### Clear History

```python
from infrastructure.chat_history import clear_session_history

# Clear specific session
success = clear_session_history("user_123")
```

## Architecture

```
┌─────────────────────────────────────────┐
│         Application Layer               │
│  (Services, Agents, API Handlers)       │
└─────────────────┬───────────────────────┘
                  │
                  ├─── getChatHistory(session_id)
                  │
┌─────────────────▼───────────────────────┐
│      Chat History Store                 │
│   (infrastructure/chat_history/)        │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  Backend Selection Logic           │ │
│  │  - Check config.backend            │ │
│  │  - Test Redis connection           │ │
│  │  - Fallback to file if needed      │ │
│  └────────────────────────────────────┘ │
└─────────────────┬───────────────────────┘
                  │
          ┌───────┴────────┐
          │                │
┌─────────▼──────┐  ┌──────▼─────────────┐
│ Redis Backend  │  │  File Backend      │
│ (Production)   │  │  (Dev/Fallback)    │
│                │  │                    │
│ - Fast         │  │ - Simple           │
│ - Distributed  │  │ - Local only       │
│ - TTL support  │  │ - JSON files       │
└────────────────┘  └────────────────────┘
```

## Production Deployment

### Docker Compose (Already Set Up)

Your `compose.dev.yaml` already includes Redis:

```yaml
services:
  redis:
    image: redis:latest
    ports:
      - "6379:6379"
    volumes:
      - redis-volume:/data
```

### Kubernetes

```yaml
# redis-deployment.yaml
apiVersion: v1
kind: Service
metadata:
  name: redis
spec:
  ports:
  - port: 6379
  selector:
    app: redis
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
```

Set environment variables:

```bash
REDIS_HOST=redis.default.svc.cluster.local
REDIS_PORT=6379
```

### Cloud Redis (AWS ElastiCache, GCP Memorystore, etc.)

Update environment variables:

```bash
REDIS_HOST=your-elasticache-endpoint.cache.amazonaws.com
REDIS_PORT=6379
REDIS_PASSWORD=your-secure-password
```

## Performance

- **Redis**: ~10,000 ops/sec, <1ms latency
- **File**: ~100 ops/sec, ~10ms latency

## Session TTL

Sessions automatically expire after configured TTL (default: 1 hour).

```yaml
chat_history:
  ttl: 3600  # 1 hour
  # ttl: 86400  # 24 hours
  # ttl: 604800  # 7 days
```

## Monitoring

Check Redis health:

```bash
# Via Docker
docker exec -it arc-fusion-redis-1 redis-cli ping
# Should return: PONG

# List all chat history keys
docker exec -it arc-fusion-redis-1 redis-cli KEYS "chat_history:*"

# View specific session
docker exec -it arc-fusion-redis-1 redis-cli LRANGE chat_history:session_id 0 -1
```

## Migration from Old System

The old file-based `PersistentChatHistory` has been replaced. No migration needed - new system will start fresh.

If you need to migrate old sessions:

```python
import json
from pathlib import Path
from infrastructure.chat_history import get_session_history

# Migrate old session
old_file = Path("./data/chat_history/old_session.json")
if old_file.exists():
    with open(old_file) as f:
        old_messages = json.load(f)

    new_history = get_session_history("old_session")
    for msg in old_messages:
        if msg['type'] == 'human':
            new_history.add_user_message(msg['content'])
        elif msg['type'] == 'ai':
            new_history.add_ai_message(msg['content'])
```

## Troubleshooting

**Redis connection failed, using file fallback**
- Check Redis is running: `docker ps | grep redis`
- Check environment variables in `.env.dev`
- Check Redis logs: `docker logs arc-fusion-redis-1`

**Messages not persisting**
- Check backend config in `config.dev.yaml`
- Check Redis connection with `redis-cli ping`
- Check application logs for errors

**High memory usage in Redis**
- Reduce TTL value
- Limit message history in agents (only use last N messages)
- Consider message size limits
