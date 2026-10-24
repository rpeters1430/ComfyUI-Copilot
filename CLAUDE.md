# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ComfyUI-Copilot is a **ComfyUI custom node package** (installed into `ComfyUI/custom_nodes/`). It registers no real nodes (`NODE_CLASS_MAPPINGS = {}`). It adds a Python backend that mounts HTTP routes on ComfyUI's `PromptServer`, plus a React sidebar UI that loads inside the ComfyUI frontend. It cannot run standalone: backend modules import ComfyUI's `server`, `folder_paths`, `nodes` and `execution`, so outside tests they only load inside a running ComfyUI process.

## Commands

Backend (run from the ComfyUI Python env):
```bash
pip install -r requirements.txt     # Windows portable: python_embeded\python.exe -m pip install -r ...
```
Restart ComfyUI to pick up backend changes. Most install failures are dependency conflicts: `openai` / `openai-agents` version clashes, or the legacy `agents` package shadowing `openai-agents`. `assets/FAQ.md` has the fixes.

Tests (no ComfyUI needed; `tests/conftest.py` stubs `server`, `folder_paths`, `nodes`, `execution`):
```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q                                        # whole suite
python -m pytest -q tests/unit/test_mcp_client.py -k merge # one file / one test
python -m ruff check .                                     # bug-class rules only, see pyproject.toml
```
`tests/unit/` holds pure-function tests. `tests/integration/` drives real routes: `test_chat_e2e.py` runs `/api/chat/invoke` through the openai-agents `Runner` against `tests/fixtures/fake_llm.py`, a fake OpenAI-compatible server that streams or returns errors. `test_plugin_load.py` loads the plugin the way ComfyUI does, in a subprocess, with and without `openai-agents`. The repo-root `__init__.py` only registers with ComfyUI when imported as a package (`if __package__:`); keep that guard, or pytest can't collect the repo.

Frontend (from `ui/`):
```bash
npm install
npm run build        # vite build -> ../dist/copilot_web, then scripts/post-build.js
npm run typecheck    # tsc --noEmit
npm run lint         # eslint; errors fail, warnings (any/unused) are tolerated legacy debt
npm test             # vitest (jsdom); one file: npx vitest run src/apis/workflowChatApi.test.ts
npx vite build --mode development   # watch mode; also spawns the Tailwind watcher (see vite.config.ts)
```
**`dist/copilot_web/` is committed and must match the source.** Never hand-edit it. ComfyUI serves the bundle from there, so a UI change does nothing until you run `npm run build` and commit `dist/`. CI fails if the committed `dist/` differs from a fresh build. Use plain `npm run build` for the fork. `build:pre` and `build:prod` bake in the dead hosted server's URL. Builds are byte-reproducible across Windows and Linux, which depends on `.gitattributes` forcing LF line endings. `post-build.js` rewrites dynamic-import paths so the bundle works when ComfyUI is served under a URL prefix. Prettier formatting is not enforced (`npm run format` exists), so don't reformat whole files in unrelated changes.

CI (`.github/workflows/ci.yml`) runs backend checks on Linux and Windows with Python 3.10 and 3.12, frontend checks plus the `dist/` sync check, and a non-blocking `pip-audit`/`npm audit`. Dependabot covers pip, npm and Actions. `ui/.npmrc` and the lockfile use `registry.npmjs.org`.

Releases: bumping `version` in `pyproject.toml` on `main` triggers `.github/workflows/publish.yml`, which publishes to the Comfy Registry.

## Architecture

### Load path
1. `__init__.py` first makes sure the `agents` import resolves to the **`openai-agents`** package, not the unrelated legacy RL package named `agents`. It then imports each controller in `backend/controller/` separately (conversation, llm, expert, provider, discovery); they register routes through `@server.PromptServer.instance.routes.*` decorators, and serves `dist/copilot_web` at `/copilot_web/`.
2. `WEB_DIRECTORY = "entry"`: ComfyUI loads `entry/entry.js`, which dynamically imports `/copilot_web/input.js` (the Vite bundle).
3. `ui/src/main.tsx` waits for the ComfyUI `app`, then registers a sidebar tab with `app.extensionManager.registerSidebarTab`. `/scripts/app.js` and `/scripts/api.js` are externals supplied by ComfyUI at runtime. In dev mode a Vite plugin rewrites them to `http://127.0.0.1:8188`.

### Local backend vs. the (suspended) hosted server
- **Local plugin routes** use relative URLs such as `/api/chat/invoke`, `/api/debug-agent`, `/api/param_debug/generate_sd_prompts`, `/api/save-workflow-checkpoint`, `/api/model_config` and `/api/expert/*`, plus the model download endpoints. They're defined in `backend/controller/*.py`.
- **Hosted Copilot server** calls are `${BASE_URL}/api/...` in the UI. That server is suspended. `config.remoteServerEnabled` in `ui/src/config.ts` is true only for builds with `VITE_API_BASE_URL`. When it's false (the default build), everything goes to the local backend: intent chats, debug, GenLab prompt writing, and chat images (sent as data URLs). Tracking, announcements and the signup/Copilot-key UI are disabled. LLM settings travel to the local backend only, through `addLocalLlmHeaders`. Send `Openai-Base-Url` even when there's no API key, because keyless local servers need it.

### Agent system (`backend/service/`)
Built on `openai-agents`, which is forced into chat-completions mode in `backend/core.py` and `agent_factory.py`.
- `agent_factory.create_agent(config=..., **agent_kwargs)` is the single place where an `Agent` is built. Endpoint, key and model resolution, plus `max_tokens` vs `max_completion_tokens` (`token_limit_param`), live in `utils/llm_settings.py`. That module is agents-free so plain-completion routes work without the SDK. A missing key becomes a placeholder, because the OpenAI client rejects empty keys and keyless local servers don't need one. `utils/llm_errors.describe_llm_error` turns 401, 404, connection and timeout errors into actionable messages.
- `conversation_api` imports the agent services inside `try`. If `openai-agents` is broken, `AGENTS_UNAVAILABLE` makes chat and debug reply with install instructions while the other routes keep working. The root `__init__.py` also loads each controller separately.
- `mcp_client.comfyui_agent_invoke` drives `/api/chat/invoke`. It builds the main "ComfyUI-Copilot" router agent with local tools (`get_current_workflow`, `search_node_local`), plus a handoff to the **Workflow Rewrite Agent** (`workflow_rewrite_agent.py` + `workflow_rewrite_tools.py`). MCP SSE servers are **optional**: none are configured by default, since the hosted upstream server is suspended. `COPILOT_REMOTE_MCP_URL` supplies `recall_workflow`/`gen_workflow`, and `COPILOT_SEARCH_MCP_URL` adds web search. `connect_mcp_servers` connects each one separately and skips any that are unreachable. `build_router_instructions` only mentions tools that actually exist, because a prompt that names a missing tool makes the model call it. A run is retried only when no text has streamed and no tool has run yet; otherwise text would be duplicated and tool side effects re-applied. It yields `(accumulated_text, {"data": ext, "finished": bool})` tuples. The controller streams these as newline-delimited `ChatResponse` JSON. `ext` is sent only in the final chunk.
- `debug_agent.py` backs `/api/debug-agent` (one-click workflow debug). It runs the workflow through `utils/comfy_gateway.py`, which talks to the local ComfyUI server. `link_agent_tools.py` and `parameter_tools.py` hold its tools.
- `message_memory.py` compresses chat history per session before each run, called via `asyncio.to_thread` because it may call the LLM synchronously. `summary_agent.py` produces the summaries and returns `None` on failure, in which case compression is skipped.
- Blocking LLM calls must not run on ComfyUI's event loop. Wrap sync OpenAI clients in `asyncio.to_thread` (see `update_workflow`), and use aiohttp with timeouts for HTTP (see `llm_api.fetch_models`).

### Request context and config
- Per-request state (session_id, workflow_checkpoint_id, config, rewrite context) is stored in `contextvars` (`utils/request_context.py`). Tools and agents read it from there instead of taking parameters. Controllers must call `set_request_context(...)` before invoking agents.
- Controllers build the config with `utils/request_utils.build_request_config`. Precedence is **request headers/body > `.env` > code defaults** (`apply_llm_env_defaults`). The `.env` file at the repo root is loaded by `utils/globals.py`; every variable is documented in `.env.example`.
- Model resolution lives in `utils/llm_settings.resolve_llm_settings`: `model_select`, then the call-site model, then `CC_DEFAULT_MODEL`, else it raises `LLMConfigurationError`, which is shown to the user. Workflow agents get their config from `key_utils.workflow_config_adapt`, which maps `workflow_llm_*` over the chat settings and keeps the chat model when no workflow model is set. Nothing defaults to the dead upstream server.
- Credentials are request-scoped. The `Authorization` bearer token goes into `config["copilot_api_key"]`, and `get_comfyui_copilot_api_key()` reads the request context. Never put per-user values in `GlobalState`. Language is a contextvar too. Log configs through `key_utils.redact_config`.
- openai-agents tracing, which uploads prompts and workflows to OpenAI, is off unless `COPILOT_ENABLE_TRACING=1`.

### Persistence
- Backend: SQLite via SQLAlchemy 1.4 (`<2.0`). The DAOs in `backend/dao/` are module-level singletons writing to `backend/data/` (`*.db` is git-ignored). The databases are `workflow_debug.db` (workflow checkpoints), `session_message.db` (per-session history plus the compression summary and index) and `rewrite_expert.db` (user-defined "expert experience" for rewrites). Tests swap a singleton for a temp DB with `monkeypatch`; see the `temp_session_db` fixture.
- Frontend: chat messages live client-side (Dexie/IndexedDB and localStorage). The full OpenAI-format history, including images, is sent with every `/api/chat/invoke` call.

### Frontend layout (`ui/src/`)
- `App.tsx` hosts two tabs: the chat (`workflowChat/`, `components/chat/`) and **GenLab** parameter tuning (`components/debug/`, see its README).
- `ChatContext.tsx` holds chat state. `utils/comfyapp.ts` wraps the ComfyUI `app` and graph APIs. `utils/comfyuiWorkflowApi2Ui.ts` converts API-format workflows into UI format before they are loaded onto the canvas.
- i18n strings live in the root `locales/{en,zh}/`. Many code comments and some docs are in Chinese. `README_CN.md` mirrors `README.md`, so update both when changing user-facing docs.

### LLM providers and capability negotiation
- **Inference** always goes through the provider's OpenAI-compatible endpoint. `backend/providers/` only adds **discovery**:
  - `OpenAICompatibleProvider` uses `/v1/models`.
  - `OllamaProvider` uses `/api/tags`, then `/api/show` for `capabilities` and `<arch>.context_length`, and `/api/ps` for the context of loaded models.
  - `LMStudioProvider` tries `/api/v1/models` (vision and `trained_for_tool_use` flags), then `/api/v0/models`, then `/v1/models`. It only falls back on 404 or unexpected shapes; a down server fails once.
  - Capabilities are `True`/`False`/`None`, where `None` means unknown and is treated permissively.
- **Registry** (`registry.py`): `detect_kind` maps port 11434 to Ollama and 1234/1235 to LM Studio; an explicit `Llm-Provider` header wins. `discover_local` probes the default ports. `lookup_model` uses a 60-second cache and never raises. The routes are `/api/providers`, `/api/providers/discover` and `/api/providers/test` in `controller/provider_api.py`. `/api/model_config` reports real `image_enable`, `tools` and `context_length`.
- **Negotiation** (`service/negotiation.py`, called at the start of `comfyui_agent_invoke`):
  - Images go to a chat model only if it can see them. Otherwise `vision_llm_model` describes them, or they're replaced with a note.
  - `resolve_tool_mode` picks the agent router, or `service/json_fallback.py` when the model reports no tool calling. `Copilot-Tool-Mode`/`COPILOT_TOOL_MODE` override it.
  - The JSON fallback makes one `complete_json` decision (answer or `rewrite_workflow`), then uses `rewrite_workflow_simple`. It passes the canvas with input names and current values, because small models otherwise rename inputs (a live 7B run turned KSampler's `cfg` into `cfg_scale`).
- `utils/structured_output.complete_json` tries `json_schema`, then `json_object`, then extracts JSON from plain text, and validates with Pydantic.
- **Every canvas rewrite goes through `workflow_rewrite_tools.apply_workflow_update`**, which runs `utils/workflow_validation.validate_api_workflow` before saving. It checks dangling links and, using `/object_info`, uninstalled node types, missing or unknown inputs, link types, combo values and number ranges. Every two-element list counts as a link, as in ComfyUI. Invalid output is rejected and the saved workflow and canvas stay unchanged. The JSON fallback retries once, feeding the problems back to the model.

### Local discovery (`backend/discovery/`)
What this ComfyUI actually has, so the LLM picks from real names instead of inventing them. Modules here must not import `openai-agents`; the agent tools wrapping them live in `service/discovery_tools.py` and `workflow_rewrite_tools.py`.
- `nodes.py`: `/object_info` through `comfy_gateway`, cached for 30 seconds (`get_node_definitions`). A failed refresh keeps the last good copy. It also holds `summarize_node` (a compact spec: input types/options/min/max plus typed outputs), `search_nodes` and `find_class` (case- and display-name-tolerant). `search_node_local`, `get_node_infos` and `get_node_info` use it and return suggestions for names that aren't installed.
- `specs.py`: ComfyUI-free helpers for input specs. It handles both combo formats (`[[options]]` and `["COMBO", {"options": [...]}]`) and `types_compatible`, where `*` and comma-separated sets work like in ComfyUI. The validator imports it.
- `models.py`: model files via `folder_paths.get_filename_list`, with names relative as the loader nodes expect. The `search_models` tool exposes it.
- `workflows.py`: the workflow library. It indexes ComfyUI's `user/default/workflows` plus `COPILOT_WORKFLOW_DIRS` only, re-parsing a file only when its mtime or size changes. It understands UI format (subgraph contents included; notes, reroutes and subgraph instances aren't node types) and API format, and finds the models each workflow uses. Entries are addressed by a path-hash id; absolute paths never leave the backend.
- `library.py`: `recall_local_workflows` returns cards (`source: "Local library"`) plus missing nodes and models. The local `recall_workflow` tool is added to the router only when no MCP server provides one. `parse_tool_output` accepts its top-level `{answer, data}`. `merge_workflow_results` waits for `gen_workflow` only when a generator is connected. In the UI, `WorkflowOption.tsx` loads library cards from `/api/discovery/workflows/{id}` instead of the hosted optimizer.
- The routes are `/api/discovery/nodes|models|workflows`, `/api/discovery/workflows/{id}` and `/api/discovery/workflows/{id}/thumbnail` (a same-named `.png`/`.webp` next to the file). Tests use `tests/fixtures/comfy_env.install()` for fake definitions, models and workflow dirs. `fake_llm.tool_call(...)` makes the fake model call a tool.
- The validator (`utils/workflow_validation.py`) also checks link types against the source node's outputs, output indexes, combo values (so a model file that isn't installed is rejected) and INT/FLOAT values and ranges.

### Provider settings UI
- `ui/src/components/providers/` has `ProviderSettings.tsx` (provider, Detect local, Test connection, workflow/vision model, tool mode) and `providerStorage.ts` (localStorage plus migration that infers the provider from an old `openaiBaseUrl`). Keep `providerStorage.ts` named differently from `ProviderSettings.tsx`: on Windows the two names collide case-insensitively.
