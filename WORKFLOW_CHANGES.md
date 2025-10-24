# RAG + Orchestration Workflow Implementation

## Overview

This document describes the new two-stage workflow that separates RAG evaluation from web search orchestration, implementing the pattern you described in your diagram.

## Architecture Changes

### Previous Flow (❌ Old)
```
Router → Planner → Tool Executor → Synthesizer → Reflection
                ↑                                      ↓
                └──────────── (retry) ─────────────────┘
```

**Problems:**
- RAG not always executed first
- No evaluation of RAG-only sufficiency
- Single reflection for both RAG and orchestration

### New Flow (✅ Current)
```
                        ┌─────────────┐
                        │    START    │
                        └──────┬──────┘
                               │
                        ┌──────▼──────┐
                        │   Router    │
                        └──────┬──────┘
                               │
                    ┌──────────┴──────────┐
              Ambiguous              Clear Question
                    │                     │
            ┌───────▼────────┐            │
            │ Clarification  │            │
            └───────┬────────┘            │
                    │                     │
         ┌──────────┼──────────┐          │
    SmallTalk  MoreDetail  Process        │
         │          │         │            │
        END        END   ┌────▼────┐       │
                        │ Refined │       │
                        │  Query  │       │
                        └────┬────┘       │
                             └────────────┘
                                  │
                    ╔═════════════▼═══════════════╗
                    ║    FIXED PART (RAG)         ║
                    ║                              ║
                    ║    ┌──────────────────┐     ║
                    ║    │   RAG Pipeline   │     ║
                    ║    │  (Retrieve +     │     ║
                    ║    │   Synthesize)    │     ║
                    ║    └────────┬─────────┘     ║
                    ║             │                ║
                    ║    ┌────────▼─────────┐     ║
                    ║    │ RAG Reflection   │     ║
                    ║    │ "Is RAG enough?" │     ║
                    ║    └────────┬─────────┘     ║
                    ╚═════════════│═══════════════╝
                                  │
                         ┌────────┴────────┐
                    Sufficient        Insufficient
                         │                 │
                        END      ╔═════════▼════════════╗
                                 ║  DYNAMIC PART        ║
                                 ║  (Orchestration)     ║
                                 ║                      ║
                                 ║  ┌───────────────┐  ║
                                 ║  │    Planner    │◄─┼──┐
                                 ║  │ (Web queries) │  ║  │
                                 ║  └───────┬───────┘  ║  │
                                 ║          │          ║  │
                                 ║  ┌───────▼───────┐  ║  │
                                 ║  │ Tool Executor │  ║  │
                                 ║  │ (Web workers) │  ║  │
                                 ║  └───────┬───────┘  ║  │
                                 ║          │          ║  │
                                 ║  ┌───────▼───────┐  ║  │
                                 ║  │  Synthesize   │  ║  │
                                 ║  │ (Merge RAG+Web│  ║  │
                                 ║  └───────┬───────┘  ║  │
                                 ║          │          ║  │
                                 ║  ┌───────▼────────┐ ║  │
                                 ║  │ Orchestration  │ ║  │
                                 ║  │  Reflection    │ ║  │
                                 ║  │ "Is complete?" │ ║  │
                                 ║  └───────┬────────┘ ║  │
                                 ╚══════════│══════════╝  │
                                            │              │
                                   ┌────────┴────────┐     │
                              Complete          Incomplete │
                                   │                  │    │
                                  END                 └────┘
                                                    (Re-plan)
```

## Key Components

### 1. New Agents

#### `RAGReflectionAgent` (agent/rag_reflection_agent.py)
**Purpose:** Evaluates if RAG results alone are sufficient to answer the query.

**Decision Criteria:**
- ✅ **Sufficient**: Documents contain all needed info, answer is complete
- ❌ **Insufficient**: Missing info, outdated data, or explicit web search needed

**Example Sufficient:**
```
Query: "What is DIN-SQL according to the paper?"
Documents: 5 relevant from research paper, confidence 0.85
Decision: SUFFICIENT → END
```

**Example Insufficient:**
```
Query: "What are the latest text-to-SQL developments this month?"
Documents: Old papers from 2022
Decision: INSUFFICIENT → Orchestration
```

#### `OrchestrationReflectionAgent` (agent/orchestration_reflection_agent.py)
**Purpose:** Evaluates final answer quality after web search supplemented RAG.

**Decision Criteria:**
- ✅ **Complete**: Answer addresses full query with RAG + Web evidence
- ❌ **Incomplete**: Still missing info, needs different search strategy

**Retry Logic:**
- Suggests different search queries
- Maximum 3 planning attempts before accepting answer

### 2. Updated Workflow Nodes

#### `rag_pipeline` Node
Executes the complete RAG workflow:
1. **RAG Retrieval** - Fetch documents from vector DB
2. **Confidence Evaluation** - Score relevance and confidence
3. **Synthesis** - Generate answer from RAG results only

#### `orchestration_synthesis` Node
Merges RAG and Web search results:
- Combines RAG documents + Web search results
- Creates comprehensive answer with citations from both sources

### 3. Workflow State Updates

**New Fields:**
```python
# RAG Reflection (first stage)
is_rag_sufficient: bool
rag_quality_score: float
rag_reflection_reasoning: str
needs_orchestration: bool

# Orchestration Reflection (second stage)
is_answer_sufficient: bool
answer_quality_score: float
reflection_issues: list
reflection_suggestions: list
```

### 4. Updated Planner

**Previous Role:** Decide between RAG/Web/Hybrid
**New Role:** Plan web search strategies to supplement RAG

**Context Changes:**
- Initial planning: "RAG was insufficient, plan web searches"
- Replanning: "Previous web search didn't help, try different strategy"

## Configuration

### New Config Entries (config.dev.yaml)

```yaml
rag_reflection_agent:
  model: gpt-4o-mini
  api_provider: openai
  temperature: 0.0
  api_key: ${OPENAI_API_KEY}
  max_tokens: 500

orchestration_reflection_agent:
  model: gpt-4o-mini
  api_provider: openai
  temperature: 0.0
  api_key: ${OPENAI_API_KEY}
  max_tokens: 800
  max_synthesis_attempts: 3
```

## Example Flows

### Scenario 1: Simple RAG Query
```
User: "What is DIN-SQL methodology?"

1. Router → Clear Question
2. RAG Pipeline → Retrieve 5 docs, synthesize answer
3. RAG Reflection → SUFFICIENT (quality: 0.9)
4. END

Result: Answer from RAG only, no web search
```

### Scenario 2: Time-Sensitive Query
```
User: "What did OpenAI release this month?"

1. Router → Clear Question
2. RAG Pipeline → No relevant docs found
3. RAG Reflection → INSUFFICIENT (needs current data)
4. Planner → Plan web search for "OpenAI releases"
5. Tool Executor → Execute web search
6. Orchestration Synthesis → Merge results
7. Orchestration Reflection → COMPLETE
8. END

Result: Answer from web search
```

### Scenario 3: Hybrid Query with Re-planning
```
User: "Explain DIN-SQL and find recent papers by the authors"

1. Router → Clear Question
2. RAG Pipeline → Find DIN-SQL info, synthesize
3. RAG Reflection → INSUFFICIENT (missing author publications)
4. Planner → Plan web search for authors
5. Tool Executor → Execute web search
6. Orchestration Synthesis → Merge RAG + Web
7. Orchestration Reflection → INSUFFICIENT (missing details)
8. Planner (retry) → Plan more specific searches
9. Tool Executor → Execute refined searches
10. Orchestration Synthesis → Merge all results
11. Orchestration Reflection → COMPLETE
12. END

Result: Answer combining RAG + multiple web searches
```

## Benefits

### 1. **Always Try RAG First**
- Every query goes through RAG pipeline
- No wasted web searches for doc-answerable queries
- Faster responses for knowledge base queries

### 2. **Clear Separation of Concerns**
- **RAG Reflection**: "Do we have the info?"
- **Orchestration Reflection**: "Is the final answer good enough?"

### 3. **Intelligent Fallback**
- RAG insufficient → Web search
- Web search insufficient → Different strategy
- Maximum 3 attempts to prevent infinite loops

### 4. **Better Context for Planning**
- Planner knows RAG already tried
- Focus on finding supplementary info
- More targeted web searches

### 5. **Efficient Resource Usage**
- RAG-only queries: 1 synthesis (cheap)
- Orchestration queries: RAG + Web (necessary)
- No redundant RAG calls after web search

## Files Modified

### New Files
- ✅ `agent/rag_reflection_agent.py`
- ✅ `agent/orchestration_reflection_agent.py`
- ✅ `prompts/rag_reflection_agent_prompt.py`

### Modified Files
- ✅ `domain/enums/llm_type.py` - Added new agent types
- ✅ `domain/enums/workflow_state.py` - Added reflection state fields
- ✅ `infrastructure/llm/workflow_graph.py` - Restructured workflow
- ✅ `prompts/planner_context_templates.py` - Updated for web-only focus
- ✅ `core/config/config.dev.yaml` - Added new agent configs
- ✅ `prompts/__init__.py` - Added RAG reflection prompt

## Testing

### Test Cases to Verify

1. **RAG-Only Query**
   - Query: "What is X from the docs?"
   - Expected: RAG sufficient, no orchestration

2. **Web-Only Query**
   - Query: "What's the latest news on Y?"
   - Expected: RAG insufficient, web search triggered

3. **Hybrid Query**
   - Query: "Compare our docs on X with latest Y developments"
   - Expected: RAG + Web orchestration

4. **Re-planning**
   - Query: Complex multi-part question
   - Expected: Multiple planning attempts if needed

### How to Test

```bash
# Start the API server
python main.py

# Send test queries via API
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test", "message": "What is DIN-SQL?"}'
```

Check logs for workflow progression:
```
[RAG Pipeline] Starting RAG retrieval and synthesis
[RAG Reflection] Evaluating RAG sufficiency
[WorkflowGraph] RAG results sufficient - ending workflow
```

Or:
```
[RAG Pipeline] Starting RAG retrieval and synthesis
[RAG Reflection] Evaluating RAG sufficiency
[WorkflowGraph] RAG results insufficient - moving to orchestration
[PlannerAgent] Creating initial execution plan
[ToolExecutor] Executing web search
[Orchestration Synthesis] Merging RAG + Web results
[Orchestration Reflection] Evaluating final answer quality
```

## Migration Notes

### Breaking Changes
- Old `reflection_agent` config renamed to `orchestration_reflection_agent`
- Workflow state has new required fields for RAG reflection
- Planner now assumes RAG already executed

### Backward Compatibility
- Tool executor still supports all tool types
- Synthesizer works for both RAG-only and orchestration
- Existing agents unchanged

## Next Steps

1. ✅ Test with sample queries
2. ✅ Monitor RAG reflection decisions (tune thresholds if needed)
3. ✅ Adjust orchestration reflection quality thresholds
4. ✅ Fine-tune planner prompts for better web query generation
5. ✅ Add metrics/logging for stage transition tracking

## Summary

The new workflow implements a **two-stage reflection pattern**:

1. **Stage 1 (RAG Reflection)**: "Is RAG sufficient?"
   - Yes → END (fast path)
   - No → Orchestration

2. **Stage 2 (Orchestration Reflection)**: "Is final answer complete?"
   - Yes → END
   - No → Re-plan (max 3 times)

This architecture ensures RAG is always tried first, web search is only used when necessary, and the system can iteratively improve answers through intelligent replanning.
