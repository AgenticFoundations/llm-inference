# On-demand LLM inference

A Python application managed with **uv**, controlled through **make**, using
your **existing native Ollama server** for Apple Silicon inference and **OrbStack**
for gateway containers. This project never launches or stops Ollama, installs another
Ollama, or changes its login settings. Downloaded model files stay in its existing store.
Generated gateway and Nginx configuration files are copied into stopped containers
through the Docker API before startup. No host folder bind mounts are used; OrbStack
does not need ongoing access to this project's folder in Documents.

```text
Local VS Code / Copilot Chat
    │ Client API key
    ▼
Nginx :4080 → LiteLLM :4000 → existing Ollama :11434 → selected model
                   │
                Postgres (client keys)
```

## First use

Prerequisites: macOS, `uv`, native `ollama`, and OrbStack. Open OrbStack once to
finish its setup and create its `orbstack` Docker context. The controller finds
OrbStack's bundled Docker/Compose tools even if they are absent from `PATH`.
It selects that context for its commands without changing your global context.

Keep your installed Ollama application/server running on `OLLAMA_PORT` (default 11434).
The controller checks its API and the selected local model. It never starts a server,
quits the application, kills an Ollama process, or changes server environment settings.
If Ollama is unavailable, startup reports an error so you can open it yourself.

```bash
make setup       # Install locked dependencies; generate private .env credentials
# Edit .env if needed, especially the exact OLLAMA_MODEL tag.
make doctor      # Check OrbStack and configuration; starts no application services
make up          # Start gateway, database, and ingress using existing Ollama
make key         # Display the default client key
make check       # Exercise auth and inference, including streaming
make down        # Stop gateway containers and unload selected model; Ollama stays running
```

`make setup` starts nothing and preserves an existing `.env`. The first `make up`
downloads container images as needed, but never pulls a model. It checks that
`OLLAMA_MODEL` exists before starting containers. The initial default is the tag
you supplied, `qwen3.8:27b-mxfp8`; use the exact installed tag if it differs.

Configuration precedence is exported shell variables, then `.env`, then defaults
from `.env.example`. For example, an exported `POSTGRES_PASSWORD` overrides the
value generated in `.env`; unset it to use the file's value. Keep this in mind
when diagnosing database authentication failures.
An exported empty value also overrides the file and fails validation for required
settings; unset the variable to fall back to `.env`.

The model loads on the first inference request, including `make check`, and
unloads after `OLLAMA_KEEP_ALIVE` of inactivity (default five minutes).
`make up`, `make doctor`, and `make status` do not warm the model.

## Commands

| Command | Purpose |
| --- | --- |
| `make` | Show commands |
| `make setup` | Install dependencies and generate `.env` once |
| `make doctor` | Check configuration, native Ollama, and OrbStack |
| `make up` | Start LLM containers connected to existing Ollama |
| `make down` | Stop LLM containers and unload only the selected model |
| `make restart` | Stop and start with current `.env` |
| `make model-start` | Preload the selected model through the existing Ollama API |
| `make model-stop` | Unload only the selected model through the existing Ollama API |
| `make status` | Show session, container state, routing, and loaded models |
| `make logs` | Follow container logs |
| `make logs SERVICE=gateway` | Follow a specific container's logs |
| `make check` | Test auth, route isolation, inference, and SSE streaming |
| `make key NAME=laptop` | Create/reuse and display a scoped client key |
| `make revoke NAME=laptop` | Revoke a client key |
| `make test` | Run tests without Ollama, containers, or model downloads |
| `make lint` | Check Python style and formatting |
| `make typecheck` | Run strict mypy checks on all application source under `src/` |

`make down` retains downloaded models, the Postgres volume, and private client-key
files. Containers have `restart: "no"`; restarting OrbStack does not restart them.
The controller never shuts down Ollama, OrbStack, or other applications' containers.
Unloading the selected model frees its memory, but other direct Ollama clients using
that exact model share its loaded state and may load it again independently. Other
loaded models are left alone. `make model-stop` can unload the selected model even
when the gateway is off; `make down` with no application session changes nothing.
Startup checks native Ollama, the exact model tag, container readiness, connectivity
from the gateway to Ollama, and authentication. It does not run inference during
startup. Failed/interrupted startup attempts roll back application resources.
Ctrl-C during `make up` triggers cleanup; Ctrl-C during
logs or checks stops that command. `make down` can stop the stack during a long
inference check.

Running `make up` again is harmless for an unchanged, healthy session. Use
`make restart` to apply `.env` edits. Shutdown uses a private snapshot of the
session configuration, so it still works if `.env` has been edited or removed.
If shutdown cannot stop the containers, it retains the session for a later
`make down` retry and does not unload a model while the gateway may still accept
new requests. If model unloading fails after container shutdown, it also retains
the session for retry. Failed startup does not unload a model that was already in
use independently. Do not delete `.runtime` while the application is running.

## Model and memory settings

```dotenv
OLLAMA_MODEL=qwen3.8:27b-mxfp8
MODEL_ALIAS=local-coder
OLLAMA_KEEP_ALIVE=5m
OLLAMA_CONTEXT_LENGTH=262144
```

Clients use `local-coder`. To switch models, set `OLLAMA_MODEL` to another locally
installed tag and run `make restart`. No client changes are needed. Context length
is sent as the per-request `num_ctx` option; keep-alive is also per request.
Increasing context length increases memory use. Parallelism and all server-wide
settings remain under your existing Ollama application's control. The former
`OLLAMA_NUM_PARALLEL` project setting is no longer used. This project disables
automatic Python interpreter downloads; `make setup` requires an existing suitable Python.

The selected MXFP8 Qwen model is served by the existing Ollama installation using
its MLX runner and safetensors weights. It reports a maximum context of 262,144
tokens (256K). Copilot
is configured with the same total window, reserving 32,768 tokens (32K) for output
and allowing 229,376 tokens (224K) for input, including instructions, tools, files,
and conversation history. Reasoning and tool calling remain enabled. Large prompts
require more memory and processing time; configuring 256K does not benchmark a full
256K prompt. If you switch models, match the context to that model's reported limit
and update the client budget too.

## Local API access

Compose manages the project's own `backend` and private `database` networks.
No externally created network is required. The Nginx API proxy is the local
client entry point at `http://127.0.0.1:4080/v1`.

The API proxy permits `GET /v1/models`, `POST /v1/chat/completions`, and a
non-sensitive `/healthz`. Other paths return 404, including LiteLLM admin/key
management endpoints, docs, and raw Ollama APIs. Chat responses stream without
Nginx buffering. LiteLLM validates bearer keys on inference and discovery requests.

Native Ollama, the local administrative gateway, and ingress are published on
loopback at ports 11434, 4000, and 4080 respectively.
Postgres has no published port. The container gateway connects to native Ollama
through `host.docker.internal`; no wildcard Ollama bind is needed.

## Connect a client

| Setting | Value |
| --- | --- |
| Base URL (local) | `http://127.0.0.1:4080/v1` |
| API type | OpenAI-compatible Chat Completions |
| Model | `local-coder` |
| API key | Output of `make key NAME=laptop` |

In current VS Code: **Chat → Manage Language Models → Add Models → Custom Endpoint**.
Choose Chat Completions and configure the endpoint, key, and model. If the editor
asks for a complete request URL, use the base URL followed by `/chat/completions`.
Copilot Chat supplies the coding harness; this server supplies inference. Agent
mode depends on the chosen model's native tool-calling capabilities. Inline
autocomplete is a separate client feature.

Each client key is scoped to the configured model alias. Keep the LiteLLM master
key local; it is the administrative credential. `make key` displays a secret on
purpose. Use `make -s key NAME=laptop` if you need only the key without Make's
command echo. Local key copies live in `.runtime/keys/` with mode 0600. Revocation
removes both the gateway key and its local copy. The default `copilot` key is
created during startup for checks; other keys can be created as needed.

`make key` and `make revoke` use the running session's configuration, including
its model alias and gateway port. Editing `.env` does not change their target;
apply those edits with `make restart` first if you want keys for the new configuration.

## Project layout

```text
Makefile                       Application entry point
pyproject.toml / uv.lock        uv project and locked dependencies
src/llm_inference/
    cli.py                     CLI dispatch
    bootstrap.py               First-use configuration and credential generation
    config.py                  .env validation and gateway config generation
    controller.py              Startup/shutdown orchestration
    runtime.py                 Compose, subprocesses, lifecycle locks, and readiness helpers
    ollama.py                  Model load/unload through the existing Ollama API
    gateway.py                 Client keys and gateway protocol checks
    transport.py               Shared HTTP requests and status checks
    storage.py                 Private atomic JSON writes
compose.yaml                   OrbStack container services
config/nginx.conf              Public route allowlist and streaming proxy
tests/                         Configuration, model lifecycle, and API tests
.env                           Private configuration (ignored)
.runtime/                      Private session, generated config, logs, keys (ignored)
```

The package name remains `llm_inference` for Python imports; all application
source lives under `src/`. Use the Makefile from this checkout. The Python CLI is
also available as `uv run --locked python -m llm_inference --help`.

Image versions are explicit in `.env`; upgrades are deliberate. Preserve the
Postgres password and LiteLLM salt across restarts. Changing the password in `.env`
does not change an existing database user's password. Unit tests mock external
services; `make check` is the integration check once prerequisites are configured.

`REQUEST_TIMEOUT` is capped at Nginx's configured 900-second proxy timeout.
Nginx's read and send timeouts measure inactivity between operations, so this
does not enforce a 900-second wall-clock deadline for an entire streamed response.

References: [OrbStack networking](https://docs.orbstack.dev/docker/network),
[LiteLLM virtual keys](https://docs.litellm.ai/docs/proxy/virtual_keys),
[Ollama model lifetime](https://docs.ollama.com/faq#how-do-i-keep-a-model-loaded-in-memory-or-make-it-unload-immediately),
[VS Code custom endpoints](https://code.visualstudio.com/docs/agent-customization/language-models#add-a-custom-endpoint-model).
