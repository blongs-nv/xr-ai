<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr_ai_kit.langchain

Optional adapters between an `xr_ai_app` app and LangChain/LangGraph, a
module of the kit (import name `xr_ai_kit.langchain`). Neither `xr_ai_app`
nor `import xr_ai_kit` imports it; an app that uses LangChain depends on
`xr-ai-kit[langchain]`, the extra that brings `langchain-core` and
`langgraph`. It uses the kit's delegate job types (`GraphBackend` is a
`Delegate` back end). The samples
`app-samples/sdk-tour/08-delegate-langgraph` and `09-langchain-agent`
use it.

> **Not qualified.** `langchain-core` and `langgraph` are not in the
> repository's qualified dependency set. Qualifying them (the dependency
> manifest, `THIRD_PARTY_NOTICES.md`, the dated cutoff in `uv.toml`) is
> pending. Until then, the module is built and tested only against the
> versions the cutoff allows: langchain-core 1.7.0 and langgraph 1.2.16
> (langgraph-prebuilt 1.1.1).

## API

| Name | What |
|---|---|
| `chat_model(role, *, max_tokens=, temperature=, enable_thinking=)` | A role as a LangChain `BaseChatModel` over `role.as_llm()`, with `ainvoke`, `astream` and `bind_tools`. Tool calls come back as `AIMessage.tool_calls`. Works with `create_react_agent` and in LangGraph nodes. |
| `langchain_tools(p, *tools)` | Our `@tool` functions (or `xr_ai_tools.Tool`s) as LangChain `StructuredTool`s bound to participant `p`. |
| `from_langchain(tool)` | A LangChain `BaseTool` as an `xr_ai_tools.Tool`, for `Conversation(tools=[...])`. |
| `await messages(p, *channels)` | Our context channels as LangChain messages. |
| `stream_text(stream)` | The AI text of a LangGraph `astream(..., stream_mode="messages")` stream, or of a LangChain chunk stream, for `p.reply(...)`. |
| `GraphBackend(graph, *, input=, result=, progress=, config=)` | A compiled LangGraph graph as a `Delegate` back end. |
| `to_chat_messages`, `to_langchain_messages` | The message conversion `chat_model` uses, in both directions. |

### Behavior to know

`chat_model`
- **Async only.** A role's clients and concurrency limit belong to the app's
  event loop, so `invoke` and `stream` raise `NotImplementedError`.
- **Lazy role resolution.** Building the model calls nothing. A module-level
  `chat_model(app.role(...))` follows the role, including the fake roles of
  `app.test_session(models=...)`.
- **One trace event per call.** Each call emits one `role` trace event
  (`method="chat"`, `via="langchain"`).
- **`tool_choice` is accepted but not enforced.** The model chooses.
- **`stop` sequences raise `ValueError`.**
- **Reasoning stays out of `content`.** It goes in
  `additional_kwargs["reasoning_content"]`.
- **Consecutive system messages are merged into one**, because many chat
  templates accept only one. For example, an agent's prompt followed by the
  output of `messages(...)` becomes a single system message.
- **Some content blocks are dropped.** Text and image blocks are kept; other
  blocks are not sent.

`langchain_tools`
- **The model sees the same schema `Conversation` offers.** `p` and `turn` are
  hidden from it.
- **`p` and `turn` are injected per call.** `p` is the participant you bound
  the tools to. `turn` is the turn being answered
  (`xr_ai_app.current_turn()`), or `None` outside a turn.
- **Results are rendered the way `Conversation` renders them**, and each call
  is a `tool` trace span.
- **A failing tool raises, as LangChain tools do.** LangGraph's `ToolNode`
  decides what the model sees.

`from_langchain`
- **The model sees the tool's `tool_call_schema`.** Injected arguments are
  left out.
- **Calls run `tool.ainvoke(arguments)`.** A sync tool runs in LangChain's
  executor.
- **Only pydantic argument schemas are accepted.** A tool whose schema is a
  plain JSON-schema dict raises `TypeError`.

`messages(p, *channels, budget_bytes=None)`
- **Returns one `SystemMessage`, or `[]` when every channel is empty.** Its
  text is the context block `Conversation` puts in its system prompt: a
  `# Context` heading, then each non-empty channel under its name, in order.
- **One message rather than one per channel**, because chat templates
  generally accept one system message.

`stream_text(stream)`
- **Yields only AI text.** It skips tool-call chunks, tool messages, human
  and system messages, and reasoning blocks.
- **Accepts these stream shapes:**
  - `(message, metadata)` pairs;
  - `(mode, payload)` pairs from a list `stream_mode`;
  - subgraph namespaces;
  - plain message chunks;
  - strings.

`GraphBackend`
- **It runs the graph with `stream_mode=["updates", "custom", "values"]`.**
  `values` supplies the final state with the graph's reducers applied.
- **A custom event can be a progress update.** That needs a dict with a
  `"progress"` key, written with `get_stream_writer()`. A `Delegate` speaks
  progress, so write it for the user.
- **A finished node is a progress update only if you ask.** Pass
  `progress(node, update)`; it returns the text, or `None` to skip a node.
  Without it, finished nodes say nothing: node names are internal and must
  not reach the user.
- **The final state becomes the result.** `result(state)` returns a string,
  or a `JobUpdate` when you want to set `speak`. The default result is:
  - the last message's text, for a message graph;
  - otherwise `state["answer"]`;
  - otherwise the state as JSON.
- **Cancelling the job cancels the running node** and closes the stream.

## Tests

The adapter tests are `tests/app/surface/test_langchain.py`. The test
project installs `xr-ai-kit[langchain]`, so the ordinary suite runs them; from
`tests/` with the cutoff-respecting resolver:

```bash
cd tests
uv --config-file ../uv.toml run pytest app/surface/test_langchain.py -q -p no:cacheprovider
```

They are skipped where `langchain-core` and `langgraph` are not installed.
