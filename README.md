**# HelloAgents Lab — Hand-Built Agent Framework

This repository contains a from-scratch reimplementation of the core building blocks of an LLM agent framework, done as a hands-on learning exercise while working through **Chapter 7 ("Building Your Own Agent Framework")** of a Chinese-language textbook on AI agent development. The goal was not to replace an existing framework, but to understand *why* agent frameworks are designed the way they are, by building the pieces independently before comparing them against the reference implementation ([HelloAgents](https://github.com/jjyaoao/HelloAgents)).

All code in `mini_agents/` is written independently and does not import from the installed `hello_agents` package — it only borrows the same design vocabulary (Message / Config / Agent / Tool) to make the comparison meaningful.

## Why build this instead of just using a framework?

Modern agent SDKs (Claude Agent SDK, OpenAI Agents SDK, LangGraph, etc.) intentionally keep their core loop minimal: give a model a system prompt, a set of tools, and a loop, and let it reason. Understanding *what that minimal loop actually needs* — message representation, config resolution, a common agent interface, a tool-calling protocol — is best learned by writing each piece once, deliberately, rather than only calling a packaged API.

## Project structure

```
mini_agents/
├── message.py              # Message data class (role-constrained, dict-serializable)
├── config.py                # Config loaded from environment variables
├── agent.py                  # Abstract Agent base class (Template Method pattern)
├── my_llm.py                 # HelloAgentsLLM subclass demonstrating provider extension
├── simple_agent.py            # SimpleAgent: multi-turn chat, function calling, streaming
├── react_agent.py             # ReActAgent: Thought -> Action -> Observation loop
├── reflection_agent.py        # ReflectionAgent: generate -> critique -> refine loop
├── plan_and_solve_agent.py    # PlanAndSolveAgent: plan first, then execute step by step
├── tools.py                   # Tool base class, ToolParameter, ToolRegistry, FunctionTool
├── calculator_tool.py         # Minimal calculator tool (eval-based, quick prototype)
├── my_calculator_tool.py      # Safer calculator using AST parsing instead of eval
├── my_advanced_search.py      # Mock multi-source search tool with fallback logic
├── tool_chain_manager.py      # ToolChain / ToolChainManager for fixed multi-step pipelines
├── async_tool_executor.py     # Async/parallel tool execution via a thread pool
└── test_*.py                  # One test script per component, runnable standalone
```

## Core design ideas explored

- **Message** — a small dataclass with a `Literal`-constrained `role` field, so an invalid role is caught at write time rather than silently accepted, plus a `to_dict()` that matches the OpenAI-style `messages` schema.
- **Config** — configuration resolved from environment variables in one place (`Config.from_env()`), instead of scattering defaults across the codebase.
- **Agent** — an abstract base class enforcing a single `run()` entry point (Template Method pattern), with shared history management (`add_message` / `get_history` / `clear_history`) implemented once and inherited by every concrete agent.
- **Four agent paradigms**, each solving a different problem:
  - `SimpleAgent` — plain multi-turn chat; extended here with native OpenAI function calling, streaming, and dynamic tool management.
  - `ReActAgent` — reasons step by step, deciding on the fly whether to call a tool, based on regex-parsed `Thought` / `Action` output.
  - `ReflectionAgent` — no external tools; improves its own answer through repeated self-critique.
  - `PlanAndSolveAgent` — plans all steps up front, then executes them in order, trading flexibility for lower token usage.
- **Tool system** — a `Tool` ABC with `get_parameters()` (self-describing schema) and `run()`, a `ToolRegistry` supporting both class-based and function-based registration, a `ToolChain` for fixed-order multi-tool pipelines, and an `AsyncToolExecutor` demonstrating real parallel speedup (verified: ~2s for 3 tasks run in parallel vs. ~6s run serially).
- **Safety note**: the calculator was deliberately reimplemented using AST parsing (`my_calculator_tool.py`) instead of `eval()`, to only permit a fixed whitelist of operators/functions rather than arbitrary code execution.

## Running the tests

Each component has a standalone test script. Example:

```bash
conda activate hello-agents
cd mini_agents
python test_message.py
python test_react_agent.py
python test_tool_chain_manager.py
```

## Environment setup

Create a `.env` file (not committed — see `.gitignore`) with:

```
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api-inference.modelscope.cn/v1/
LLM_MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
```

Any OpenAI-compatible endpoint works; this project was developed and tested against ModelScope's free inference API.

## Notes

- This is a learning artifact, not a production framework. Error handling, retries, and security hardening are present in places (e.g. AST-based calculator, exponential backoff for rate limits) but not applied uniformly everywhere.
- The reference framework this project is modeled after is [HelloAgents](https://github.com/jjyaoao/HelloAgents), part of the *AI Agent 开发实践* textbook.**
