# HelloAgents Lab — Hands-On Notes on Building an Agent Framework from Scratch

This repository is a hands-on learning journal for a Chinese-language textbook on AI agent development ([HelloAgents](https://github.com/jjyaoao/HelloAgents)). Rather than just reading the reference implementation, each chapter here is reimplemented independently from scratch, then compared against the book's design — the goal is to understand *why* agent frameworks are built the way they are, not just to reuse someone else's code.

All code borrows the reference project's design vocabulary (Message / Config / Agent / Tool / Memory / RAG) to make the comparison meaningful, but nothing imports from the installed `hello_agents` package directly.

## Repository structure

```
hello-agents-lab/
├── mini_agents/     # Chapter 7 — hand-built agent framework (paradigms, tools, tool-calling)
└── mini_memory/     # Chapter 8 — hand-built memory system + RAG
```

Each folder is self-contained and has its own runnable test scripts.

---

## Chapter 7 — `mini_agents/`: building an agent framework

Explores what a minimal agent loop actually needs: message representation, config resolution, a common agent interface, and a tool-calling protocol.

```
mini_agents/
├── message.py                 # Message data class (role-constrained, dict-serializable)
├── config.py                  # Config loaded from environment variables
├── agent.py                   # Abstract Agent base class (Template Method pattern)
├── my_llm.py                  # HelloAgentsLLM subclass demonstrating provider extension
├── simple_agent.py            # SimpleAgent: multi-turn chat, function calling, streaming
├── react_agent.py             # ReActAgent: Thought -> Action -> Observation loop
├── reflection_agent.py        # ReflectionAgent: generate -> critique -> refine loop
├── plan_and_solve_agent.py    # PlanAndSolveAgent: plan first, then execute step by step
├── tools.py                   # Tool base class, ToolParameter, ToolRegistry, FunctionTool
├── my_calculator_tool.py      # Safer calculator using AST parsing instead of eval
├── my_advanced_search.py      # Mock multi-source search tool with fallback logic
├── tool_chain_manager.py      # ToolChain / ToolChainManager for fixed multi-step pipelines
├── async_tool_executor.py     # Async/parallel tool execution via a thread pool
└── test_*.py                  # One test script per component, runnable standalone
```

**Four agent paradigms, each solving a different problem:**

| Paradigm | External tools? | Decision style | Best for |
|---|---|---|---|
| `SimpleAgent` | Optional (native function calling) | No step concept | Plain Q&A |
| `ReActAgent` | Yes | Dynamic, step by step | Tasks needing lookup/computation |
| `ReflectionAgent` | No (self-critique) | Fixed number of rounds | Iterative quality refinement |
| `PlanAndSolveAgent` | Optional | Static, planned up front | Multi-step tasks with a clear structure |

The calculator tool was deliberately reimplemented using AST parsing rather than `eval()`, to only permit a whitelist of operators/functions instead of arbitrary code execution. The async tool executor demonstrates a verified real speedup: ~2s for 3 tasks run in parallel vs. ~6s run serially.

```bash
cd mini_agents
python test_react_agent.py
python test_tool_chain_manager.py
```

---

## Chapter 8 — `mini_memory/`: memory system + RAG

Explores the two things a stateless LLM fundamentally lacks: persistent memory across sessions, and access to knowledge outside its training data.

```
mini_memory/
├── memory_item.py         # MemoryItem / MemoryConfig — the shared data structure
├── working_memory.py      # In-memory + TTL expiry + capacity eviction
├── episodic_memory.py     # SQLite + TF-IDF vector search
├── semantic_memory.py     # networkx graph (stands in for Neo4j) + vector search
├── perceptual_memory.py   # Modality-isolated vector stores (text/image/audio)
├── test_scoring.py        # Standalone verification of the four scoring formulas
├── memory_tool.py         # MemoryTool — unified execute(action, **kwargs) entry point
├── rag_basic.py            # Real PDF loading (pypdf) + paragraph-aware chunking + retrieval
├── rag_advanced.py         # Multi-Query Expansion (MQE) + HyDE
└── pdf_assistant.py         # PDFLearningAssistant — combines MemoryTool + RAGTool
```

**Four memory types, each with a different storage design:**

| Type | Storage | Retrieval weighting | Why |
|---|---|---|---|
| Working | Pure in-memory + TTL | keyword + vector, recency-weighted | Short-lived, needs to be fast, not durable |
| Episodic | SQLite + vector | `vector×0.8 + recency×0.2` | Concrete timestamped events — recency matters |
| Semantic | Graph (networkx) + vector | `vector×0.7 + graph×0.3` | Abstract knowledge — relationships matter more than freshness |
| Perceptual | Modality-isolated vector stores | `vector×0.8 + recency×0.2` | Different modalities have incompatible vector dimensions |

Every retrieval formula above was independently verified in `test_scoring.py` rather than just copied from the book — e.g. confirming that the graph-search weight in semantic memory (0.3) actually produces a higher score contribution than the recency weight in episodic memory (0.2), and that 2-hop graph traversal can surface a related memory that pure vector search misses entirely (zero literal overlap in the text).

**RAG pipeline**: real PDF → `pypdf` text extraction → paragraph-aware sliding-window chunking (`chunk_size`/`chunk_overlap`, avoids cutting mid-sentence) → TF-IDF vector index (char n-grams, since Chinese has no word boundaries for whitespace tokenizers) → optional Multi-Query Expansion + HyDE for queries that don't literally overlap with the source text → LLM-generated answer with cited chunk sources.

**`PDFLearningAssistant`** composes `MemoryTool` and `RAGTool` (not inheritance) into a per-user learning assistant: isolated SQLite file and RAG namespace per `user_id`, full Q&A history retained (not just printed to stdout), notes tagged by concept into semantic memory, and a JSON learning report on demand.

```bash
cd mini_memory
python memory_tool.py
python pdf_assistant.py   # point file_path at your own PDF first
```

---

## Environment setup (shared across both chapters)

Both chapters were developed and tested on an AutoDL cloud GPU instance, using ModelScope's free inference API (any OpenAI-compatible endpoint works).

```bash
pip install scikit-learn numpy python-dotenv openai networkx pypdf
```

Create a `.env` file (not committed — see `.gitignore`) in each chapter's working directory:

```
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api-inference.modelscope.cn/v1/
LLM_MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
```

## Notes

- This is a learning artifact, not a production framework. Real deployments would use Qdrant/Neo4j for the memory system's vector/graph storage and MarkItDown for PDF conversion; this repo intentionally substitutes zero-cost local equivalents (SQLite, TF-IDF, networkx, pypdf) so the core logic can be verified without any cloud service accounts.
- Error handling, retries (exponential backoff on rate limits), and security hardening (AST-based calculator instead of `eval`) are present where they mattered most for the learning goal, not applied uniformly everywhere.
- The reference framework this project is modeled after is [HelloAgents](https://github.com/jjyaoao/HelloAgents), part of the *AI Agent 开发实践* textbook.
