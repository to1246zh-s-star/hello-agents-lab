# HelloAgents Lab — Hands-On Notes on Building an Agent Framework from Scratch

This repository is a hands-on learning journal for a Chinese-language textbook on AI agent development ([HelloAgents](https://github.com/jjyaoao/HelloAgents)). Rather than just reading the reference implementation, each chapter here is reimplemented independently from scratch, then compared against the book's design — the goal is to understand *why* agent frameworks are built the way they are, not just to reuse someone else's code.

Chapters 7–9 borrow the reference project's design vocabulary (Message / Config / Agent / Tool / Memory / RAG / Context) to make the comparison meaningful, but reimplement everything independently rather than importing from the installed `hello_agents` package. Chapter 10 is different by nature — it's about *using* three interoperability protocols (MCP / A2A / ANP) that only make sense as ecosystem-level standards, so it runs directly against the installed `hello-agents[protocol]` package and community/official protocol servers instead of reimplementing them.

## Repository structure

```
hello-agents-lab/
├── mini_agents/       # Chapter 7  — hand-built agent framework (paradigms, tools, tool-calling)
├── mini_memory/       # Chapter 8  — hand-built memory system + RAG
├── mini_context/      # Chapter 9  — hand-built context engineering pipeline
└── mini_protocols/    # Chapter 10 — MCP / A2A / ANP protocol integration (uses hello-agents directly)
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

## Chapter 9 — `mini_context/`: context engineering

Explores what actually goes into the context window a model sees at sampling time — not just the prompt, but everything auto-assembled around it (tool definitions, retrieved memory, conversation history) — and how that context degrades ("context rot") as it grows.

```
mini_context/
├── context_builder.py         # ContextPacket + ContextConfig, GSSC pipeline (Gather-Select-Structure-Compress)
├── note_tool.py                # Markdown+YAML note tool: create/read/update/search/list/summary/delete
├── terminal_tool.py            # Sandboxed terminal: command whitelist + directory sandbox + timeout + output cap
├── codebase_maintainer.py      # Long-horizon agent case study, composes the three tools above + MemoryTool
├── message.py / llm_client.py  # Message class + HelloAgentsLLM wrapper (+ MockLLM for offline testing)
├── relevance_scorer.py         # TfidfContextBuilder — char n-gram TF-IDF instead of whitespace-Jaccard
├── context_quality.py          # 3-axis context quality scoring (density / relevance / completeness)
├── sub_agent.py                 # Sub-agent architecture with summaries persisted via NoteTool
├── context_router.py           # Per-role context tailoring instead of broadcasting one context to all roles
├── prompt_cache_layout.py      # Splits cacheable_prefix / dynamic_suffix for prompt-cache-friendly ordering
└── context_aware_agent_v2.py   # Integrates all of the above
```

**Real, tested findings — not just re-implementing the book's design:**

- The book's default relevance scoring (`_calculate_relevance()`) uses whitespace-split Jaccard similarity, which **silently fails on Chinese text** (no word boundaries) — Chinese memories were getting filtered out by `min_relevance` even when clearly relevant. `TfidfContextBuilder` (char n-gram TF-IDF) fixes this, verified against real Chinese queries.
- `ContextConfig.reserve_ratio` is documented as reserving budget for the system instruction, but the shipped `_select()` implementation never actually reads that field — a real doc/implementation gap, not a bug in this reimplementation.
- The book's section 9.2.3 describes long-context compaction as requiring high-fidelity LLM summarization, but the paired reference code's `_compress()` is plain character truncation — confirmed by reading both the prose and the code side by side, not assumed.

```bash
cd mini_context
python context_aware_agent_v2.py
```

---

## Chapter 10 — `mini_protocols/`: MCP / A2A / ANP protocol integration

Unlike chapters 7–9, this chapter is about interoperability protocols, not reimplementation — the whole point of MCP/A2A/ANP is to be shared standards, so this folder runs directly against `hello-agents[protocol]` and real MCP servers (both community-provided and self-written) instead of reinventing them.

```
mini_protocols/
├── mcp/
│   ├── 01_memory_transport.py         # MCPTool() with the built-in demo server (no network needed)
│   ├── 02_auto_expand.py              # SimpleAgent + MCPTool auto-expansion (6 calculator tools)
│   ├── 03_filesystem_stdio.py         # npx-launched community filesystem MCP server
│   ├── 04_github_mcp.py               # npx-launched GitHub MCP server (needs a PAT)
│   └── 05_multi_agent_doc_assistant.py# Two SimpleAgents (GitHub search + doc writer) sharing MCP tools
├── mcp_custom_server/
│   ├── weather_mcp_server.py          # Self-written MCP server (wttr.in-backed weather tool)
│   ├── test_weather_server.py         # Raw MCPClient test against the self-written server
│   └── weather_assistant.py           # SimpleAgent wired to the self-written server
├── a2a/
│   ├── 01_calculator_agent.py         # A2AServer with local skills, no networking
│   ├── 02_server_client.py            # Real A2AServer + A2AClient over localhost
│   ├── 03_agent_network.py            # 3-agent pipeline: researcher -> writer -> editor
│   ├── 04_a2a_tool_integration.py     # A2ATool wired into a SimpleAgent coordinator
│   ├── 05_customer_service.py         # 3-agent customer service system (receptionist/tech/sales)
│   └── 06_negotiation.py              # Proposal/counter-proposal negotiation between two agents
└── anp/
    ├── 01_discovery.py                 # ANPDiscovery + register_service + discover_service
    ├── 02_network.py                   # ANPNetwork node/edge construction
    └── 03_task_scheduling.py           # SimpleAgent + ANPTool choosing among 10 mock compute nodes
```

**Three protocols, three relationship types:**

| Protocol | Relationship | Identity/auth | Stateful? | Best for |
|---|---|---|---|---|
| MCP | Agent ↔ Tool | None (local/trusted server) | No (single call) | Giving one agent access to external capabilities |
| A2A | Agent ↔ Agent (peer-to-peer) | Yes (discovery + auth is part of the flow) | Yes (Task lifecycle) | Small, tightly-coupled multi-agent collaboration |
| ANP | Agent ↔ Agent network (large-scale) | Yes (DID-based) | Service-level (register/discover) | Large, open, decentralized agent networks |

**Real bugs found and root-caused during testing, not just re-running book code:**

- `a2a-sdk` shipped a breaking API change between `0.x`/`1.0.0` and `1.1.0` — `A2AClient` was removed from `a2a.client` entirely, replaced with `Client`/`ClientFactory`/`create_client`. `hello-agents[protocol]==0.2.2` targets the pre-1.0 interface, so `pip install a2a-sdk` (which grabs latest) silently breaks `A2A_AVAILABLE`. Confirmed by manually reproducing the swallowed `try/except ImportError` and diffing `dir(a2a.client)` across versions.
- `A2AServer.run()` requires Flask, which is not pulled in by either `hello-agents[protocol]` or `a2a-sdk` — a genuinely undeclared dependency, not documented anywhere in the book.
- Because `A2AServer.run()` is started in a `daemon=True` thread, a missing-Flask exception there is silently swallowed by the thread boundary — the main thread's "server started" print statement fires unconditionally regardless of whether the server actually came up, and the client call downstream returns `None` instead of raising. Replaced the book's fixed `time.sleep(2)` with an active `wait_for_port()` poll to make startup failures observable instead of silently hanging.
- The simplified `[TOOL_CALL:tool_name:key=value]` text protocol HelloAgents uses to let an LLM invoke expanded MCP tools breaks on natural-language arguments containing spaces (e.g. a GitHub search query like `"AI agent"`) — the same class of failure as the whitespace-Jaccard bug found in Chapter 9, just showing up at the tool-call-parsing layer instead of the retrieval layer.

```bash
cd mini_protocols
python mcp/02_auto_expand.py
python a2a/02_server_client.py
python anp/03_task_scheduling.py
```

---

## Environment setup (shared across all chapters)

All chapters were developed and tested on an AutoDL cloud GPU instance, using ModelScope's free inference API (any OpenAI-compatible endpoint works).

```bash
# Chapters 7-9
pip install scikit-learn numpy python-dotenv openai networkx pypdf

# Chapter 10 additionally needs
pip install "hello-agents[protocol]==0.2.2" a2a-sdk flask
# + Node.js (for npx-launched community MCP servers)
```

Create a `.env` file (not committed — see `.gitignore`) in each chapter's working directory:

```
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api-inference.modelscope.cn/v1/
LLM_MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
```

## Notes

- This is a learning artifact, not a production framework. Real deployments would use Qdrant/Neo4j for the memory system's vector/graph storage and MarkItDown for PDF conversion; chapters 7-9 intentionally substitute zero-cost local equivalents (SQLite, TF-IDF, networkx, pypdf) so the core logic can be verified without any cloud service accounts. Chapter 10 is the one exception to the "reimplement from scratch" approach, since MCP/A2A/ANP are only meaningful as shared, installed standards.
- Error handling, retries (exponential backoff on rate limits), and security hardening (AST-based calculator instead of `eval`) are present where they mattered most for the learning goal, not applied uniformly everywhere.
- The reference framework this project is modeled after is [HelloAgents](https://github.com/jjyaoao/HelloAgents), part of the *AI Agent 开发实践* textbook.
