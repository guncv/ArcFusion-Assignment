# Separate Synthesizers Implementation

## Overview

The workflow now uses **two specialized synthesizers** instead of a single generic one, providing better separation of concerns and optimized prompts for each stage.

## The Problem with Single Synthesizer

**Before (❌):**
```
Same SynthesizerAgent for both:
1. RAG-only synthesis (knowledge base documents)
2. Orchestration synthesis (RAG + Web merge)

Issues:
- Generic prompts trying to handle both cases
- Can't optimize for RAG-only vs. merging strategies
- Harder to debug which synthesis stage fails
- No clear distinction in citations and source handling
```

## The Solution: Two Specialized Synthesizers

**After (✅):**
```
1. RAG Synthesizer → RAG Pipeline stage
   - Focus: RAG documents ONLY
   - Task: Be honest about document limitations
   - Citations: Document references only

2. Orchestration Synthesizer → Orchestration stage
   - Focus: Merge RAG + Web intelligently
   - Task: Combine both sources comprehensively
   - Citations: Both documents AND web URLs
```

## Architecture

### 1. RAG Synthesizer Agent

**File:** `agent/rag_synthesizer_agent.py`
**Prompt:** `prompts/rag_synthesizer_agent_prompt.py`

**Purpose:**
- Synthesizes answers from RAG documents only
- Used in the `rag_pipeline` node
- First synthesis stage (always executed)

**Key Responsibilities:**
- ✅ Generate answer from knowledge base documents only
- ✅ Cite all claims with document references
- ✅ Be honest about information gaps
- ✅ Explicitly state what's missing from documents
- ❌ NO web search results (not available at this stage)
- ❌ NO speculation beyond documents

**Prompt Strategy:**
- Emphasizes using ONLY retrieved documents
- Encourages honesty about missing information
- Guides being specific about gaps (helps RAG Reflection decide)

**Example Output:**
```
"The DIN-SQL approach uses decomposition for text-to-SQL generation
[Source: din_sql.pdf, Page 3]. The methodology achieves 85.3% accuracy
on Spider benchmark [Source: din_sql.pdf, Page 7].

However, the retrieved documents do not contain information about the
authors' other publications or recent work."
```

**Configuration:**
```yaml
rag_synthesizer_agent:
  model: gpt-4o-mini
  temperature: 0.2  # Lower temp for factual RAG synthesis
  max_tokens: 1500
```

---

### 2. Orchestration Synthesizer Agent

**File:** `agent/orchestration_synthesizer_agent.py`
**Prompt:** `prompts/orchestration_synthesizer_agent_prompt.py`

**Purpose:**
- Merges RAG and Web search results
- Used in the `orchestration_synthesis` node
- Second synthesis stage (only if RAG insufficient)

**Key Responsibilities:**
- ✅ Intelligently merge RAG + Web information
- ✅ Prioritize sources appropriately (RAG for internal, Web for current)
- ✅ Handle conflicting information gracefully
- ✅ Cite both document and web sources
- ✅ Incorporate reflection feedback for iterative improvement

**Prompt Strategy:**
- Guides on when to prioritize RAG vs. Web
- Explains how to handle source conflicts
- Provides examples of different merging scenarios
- Emphasizes comprehensive citations from both sources

**Example Output:**
```
"The DIN-SQL methodology is described in detail in the research paper
[Source: din_sql.pdf, Pages 1-8]. The lead author, Dr. Mohammadreza
Pourreza, is an assistant professor at UC Berkeley specializing in NLP
and databases [Source: UC Berkeley Faculty - https://berkeley.edu/faculty/pourreza].

Recent benchmarks show DIN-SQL outperforms GPT-4 by 12% on complex
queries [Source: ML Benchmarks 2024 - https://benchmarks.com/sql],
demonstrating the real-world impact of the approach detailed in the
original paper."
```

**Configuration:**
```yaml
orchestration_synthesizer_agent:
  model: gpt-4o-mini
  temperature: 0.3  # Slightly higher for creative merging
  max_tokens: 2500  # More tokens for comprehensive merged answers
```

---

## Workflow Integration

### RAG Pipeline (Stage 1)
```
1. RAG Retrieval → Documents
2. Confidence Evaluation → Score
3. RAG Synthesizer ← Uses RAG-only documents
4. RAG Reflection → Evaluates sufficiency
```

**Log Output:**
```
[RAG Pipeline] Starting RAG retrieval and synthesis
[RAG Synthesizer] Synthesizing from 5 documents (confidence: 0.85)
[RAG Synthesizer] Generated answer (523 chars)
[RAG Pipeline] Complete - Retrieved: 5 docs, Confidence: 0.85
```

### Orchestration (Stage 2)
```
1. Planner → Web search queries
2. Tool Executor → Web results
3. Orchestration Synthesizer ← Merges RAG + Web
4. Orchestration Reflection → Evaluates completeness
```

**Log Output:**
```
[Orchestration Synthesis] Merging RAG (5 docs) + Web (3 results), attempt #1
[Orchestration Synthesizer] Generated merged answer (1247 chars, 8 sources)
```

---

## Comparison Table

| Aspect | RAG Synthesizer | Orchestration Synthesizer |
|--------|----------------|---------------------------|
| **Stage** | RAG Pipeline (always first) | Orchestration (if RAG insufficient) |
| **Input Sources** | RAG documents only | RAG documents + Web results |
| **Primary Goal** | Honest RAG-only answer | Comprehensive merged answer |
| **Citations** | Documents only | Documents + Web URLs |
| **Temperature** | 0.2 (factual) | 0.3 (creative merging) |
| **Max Tokens** | 1500 | 2500 |
| **Key Focus** | Identify gaps | Fill gaps |
| **Reflection** | RAG Reflection | Orchestration Reflection |
| **Retry Logic** | No retries | Up to 3 attempts |

---

## Example Scenarios

### Scenario 1: RAG Sufficient (Single Synthesizer)

**Query:** "What is DIN-SQL methodology?"

**Flow:**
1. RAG Synthesizer generates complete answer from documents
2. RAG Reflection: ✅ Sufficient
3. **END** (Orchestration Synthesizer never called)

**Result:** Fast response using only RAG Synthesizer

---

### Scenario 2: RAG Insufficient (Both Synthesizers)

**Query:** "Who created DIN-SQL and what are their recent publications?"

**Flow:**
1. **RAG Synthesizer** generates partial answer:
   - "DIN-SQL methodology is described [Source: paper.pdf]"
   - "However, documents lack author biographical info"
2. RAG Reflection: ❌ Insufficient
3. Planner plans web search for authors
4. **Orchestration Synthesizer** merges both:
   - RAG: Methodology details
   - Web: Author bios and publications
5. Orchestration Reflection: ✅ Complete

**Result:** Comprehensive answer using both synthesizers

---

### Scenario 3: Iterative Orchestration (Multiple Attempts)

**Query:** "Compare DIN-SQL with recent text-to-SQL methods"

**Flow:**
1. RAG Synthesizer: Explains DIN-SQL from docs
2. RAG Reflection: ❌ Insufficient (missing recent methods)
3. **Attempt 1:**
   - Orchestration Synthesizer: Merges RAG + Web (basic)
   - Orchestration Reflection: ❌ Incomplete (missing specific comparisons)
4. **Attempt 2:**
   - Planner: Plans more specific searches
   - Orchestration Synthesizer: Merges with improved searches
   - Uses feedback: "Add specific performance comparisons"
   - Orchestration Reflection: ✅ Complete

**Result:** Iteratively improved answer through re-synthesis

---

## Benefits

### 1. **Clearer Separation of Concerns**
- RAG Synthesizer focuses on documents
- Orchestration Synthesizer focuses on merging
- No confusion about which task to perform

### 2. **Optimized Prompts**
- RAG prompt emphasizes honesty about gaps
- Orchestration prompt emphasizes intelligent merging
- Each prompt tailored to specific task

### 3. **Better Debugging**
- Log messages clearly show which synthesizer is active
- Easy to identify if RAG synthesis or orchestration synthesis fails
- Separate error tracking for each stage

### 4. **Improved Quality**
- RAG Synthesizer better at identifying gaps (helps reflection)
- Orchestration Synthesizer better at merging sources
- Each agent specialized for its task

### 5. **Flexible Configuration**
- Different temperatures for different tasks
- Different token limits based on complexity
- Can tune each synthesizer independently

---

## Files Modified/Created

### New Files
- ✅ `agent/rag_synthesizer_agent.py`
- ✅ `agent/orchestration_synthesizer_agent.py`
- ✅ `prompts/rag_synthesizer_agent_prompt.py`
- ✅ `prompts/orchestration_synthesizer_agent_prompt.py`

### Modified Files
- ✅ `domain/enums/llm_type.py` - Added `RAG_SYNTHESIZER_AGENT`, `ORCHESTRATION_SYNTHESIZER_AGENT`
- ✅ `infrastructure/llm/workflow_graph.py` - Uses separate synthesizers in pipeline
- ✅ `core/config/config.dev.yaml` - Separate configs for each synthesizer
- ✅ `prompts/__init__.py` - Export new prompts

### Legacy Files (Kept for Compatibility)
- `agent/synthesizer_agent.py` - Original generic synthesizer (unused)
- `prompts/synthesizer_agent_prompt.py` - Original prompt (unused)

---

## Testing

### Verify RAG Synthesizer

**Test Query:** "What is DIN-SQL?"

**Expected Behavior:**
1. RAG Synthesizer generates answer from documents only
2. No web sources mentioned
3. Clear document citations
4. If info missing, explicitly states what's not in docs

**Check Logs:**
```
[RAG Synthesizer] Synthesizing from X documents
[RAG Synthesizer] Generated answer (X chars)
```

---

### Verify Orchestration Synthesizer

**Test Query:** "What are the latest text-to-SQL developments?"

**Expected Behavior:**
1. RAG Synthesizer acknowledges documents are outdated
2. RAG Reflection triggers orchestration
3. Orchestration Synthesizer merges RAG + Web
4. Both document and web citations present

**Check Logs:**
```
[RAG Synthesizer] Synthesizing from X documents
[RAG Reflection] Evaluating RAG sufficiency
[WorkflowGraph] RAG results insufficient - moving to orchestration
[Orchestration Synthesizer] Merging RAG (X docs) + Web (Y results)
[Orchestration Synthesizer] Generated merged answer (X chars, Y sources)
```

---

## Migration Notes

### Breaking Changes
None - this is additive. Old `SynthesizerAgent` still exists for compatibility.

### Backward Compatibility
- Old synthesizer code still in codebase
- New workflow uses new synthesizers
- Old imports won't break

### Configuration Updates Required
Add to `config.dev.yaml`:
```yaml
rag_synthesizer_agent:
  model: gpt-4o-mini
  temperature: 0.2
  api_key: ${OPENAI_API_KEY}
  max_tokens: 1500

orchestration_synthesizer_agent:
  model: gpt-4o-mini
  temperature: 0.3
  api_key: ${OPENAI_API_KEY}
  max_tokens: 2500
```

---

## Summary

The workflow now uses **two specialized synthesizers**:

1. **RAG Synthesizer** (Stage 1)
   - RAG documents only
   - Honest about gaps
   - Helps reflection decide if orchestration needed

2. **Orchestration Synthesizer** (Stage 2)
   - Merges RAG + Web
   - Comprehensive answers
   - Iterative improvement through reflection

This separation provides:
- ✅ Better prompts for each task
- ✅ Clearer debugging and logging
- ✅ Optimized configurations
- ✅ Improved answer quality
- ✅ Easier to maintain and tune

The workflow is more modular, each component has a clear purpose, and the system can better handle both simple (RAG-only) and complex (RAG+Web) queries.
