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
The controller checks its API and the enabled chat models in the catalog. It never starts a server,
quits the application, kills an Ollama process, or changes server environment settings.
If Ollama is unavailable, startup reports an error so you can open it yourself.

```bash
make setup       # Install locked dependencies; generate private .env credentials
# Edit config/models.json for models; .env contains service settings and credentials.
make doctor      # Check OrbStack and configuration; starts no application services
make up          # Start gateway, database, and ingress using existing Ollama
make key         # Display the default client key
make check       # Exercise auth and inference, including streaming
make down        # Stop gateway containers and unload configured models; Ollama stays running
```

`make setup` starts nothing and preserves an existing `.env`. The first `make up`
downloads container images as needed, but never pulls a model. It checks that
each enabled chat model exists before starting containers. The catalog includes
`qwen3.8:27b-mxfp8` and `qwen3.8:27b-mlx-bf16`; edit their tags to match your installed models.

Configuration precedence is exported shell variables, then `.env`, then defaults
from `.env.example`. For example, an exported `POSTGRES_PASSWORD` overrides the
value generated in `.env`; unset it to use the file's value. Keep this in mind
when diagnosing database authentication failures.
An exported empty value also overrides the file and fails validation for required
settings; unset the variable to fall back to `.env`.

The model loads on the first inference request, including `make check`, and
unloads after its catalog `keep_alive` duration of inactivity (five minutes for the Qwen entries).
`make up`, `make doctor`, and `make status` do not warm the model.

## Commands

| Command | Purpose |
| --- | --- |
| `make` | Show commands |
| `make setup` | Install dependencies and generate `.env` once |
| `make doctor` | Check configuration, native Ollama, and OrbStack |
| `make up` | Start LLM containers connected to existing Ollama |
| `make down` | Stop LLM containers and unload only configured models |
| `make restart` | Stop and start with current service settings and model catalog |
| `make model-start MODEL=local-coder` | Preload exactly the chosen chat model |
| `make model-stop` | Unload configured models through the existing Ollama API |
| `make status` | Show session, container state, routing, and loaded models |
| `make logs` | Follow container logs |
| `make logs SERVICE=gateway` | Follow a specific container's logs |
| `make check` | Test auth, routes, inference, and SSE for every enabled chat model |
| `make check MODEL=local-coder-bf16` | Test inference and SSE for only this alias |
| `make copilot-config` | Export client model entries from the catalog; leaves VS Code unchanged |
| `make key NAME=laptop` | Create/reuse and display a scoped client key |
| `make revoke NAME=laptop` | Revoke a client key |
| `make test` | Run tests without Ollama, containers, or model downloads |
| `make lint` | Check Python style and formatting |
| `make typecheck` | Run strict mypy checks on all application source under `src/` |

`make down` retains downloaded models, the Postgres volume, and private client-key
files. Containers have `restart: "no"`; restarting OrbStack does not restart them.
The controller never shuts down Ollama, OrbStack, or other applications' containers.
Unloading configured models frees their memory, but other direct Ollama clients using
those exact models share their loaded state and may load them again independently. Other
loaded models are left alone. `make model-stop` can unload configured models even
when the gateway is off; `make down` with no application session changes nothing.
Startup checks native Ollama, all enabled chat model tags, container readiness, connectivity
from the gateway to Ollama, and authentication. It does not run inference during
startup. Failed/interrupted startup attempts roll back application resources.
Ctrl-C during `make up` triggers cleanup; Ctrl-C during
logs or checks stops that command. `make down` can stop the stack during a long
inference check.

Running `make up` again is harmless for an unchanged, healthy session. Use
`make restart` to apply `.env` or model catalog edits. Shutdown uses a private snapshot
of both service settings and the complete model catalog, so it still works if either
configuration file has been edited or removed. Older saved sessions remain readable for cleanup.
If shutdown cannot stop the containers, it retains the session for a later
`make down` retry and does not unload a model while the gateway may still accept
new requests. If model unloading fails after container shutdown, it also retains
the session for retry. Failed startup does not unload a model that was already in
use independently. Do not delete `.runtime` while the application is running.

## Model catalog

All model definitions live in `config/models.json`. `.env` selects the file with
`MODELS_CONFIG=config/models.json`; the path may also be absolute. The catalog is
loaded and validated once per command. There is no primary/additional distinction
and no default model: requests choose an alias, and explicit preloading requires
`make model-start MODEL=alias`.

Each chat entry has its own configuration:

```json
{
  "alias": "local-coder",
  "name": "Local Qwen MXFP8",
  "type": "chat",
  "model": "qwen3.8:27b-mxfp8",
  "context_window": 262144,
  "max_input_tokens": 229376,
  "max_output_tokens": 32768,
  "keep_alive": "5m",
  "timeout": 900,
  "capabilities": {
    "input": ["text", "image"],
    "output": ["text"],
    "tools": true,
    "thinking": true
  }
}
```

`alias` is the ID clients request, `name` is the client display name, and `model`
is the installed Ollama tag. Input plus output budgets must fit within the context
window. The gateway sends `context_window` as Ollama's per-request `num_ctx`, and
uses this entry's keep-alive, timeout, and default output limit. Client exports use
these same budgets and capabilities. Clients can still override the output limit
in a request; declared input budgets are client guidance and gateway metadata.
Add `"enabled": false` to retain a chat definition without exposing or loading it.
Aliases are unique across all entries, including disabled ones.

The catalog contains two enabled chat models: `local-coder` for MXFP8 and
`local-coder-bf16` for BF16. Both currently use a 256K context with 224K reserved
for input and 32K for output. These are independent settings: each model can use
a different context, token budget, timeout, and keep-alive. Match the window to the
model's actual limit. Larger prompts use more memory and take longer to process.
The existing Ollama application controls global parallelism, residency, and eviction.
No model is downloaded or preloaded during startup.

The separate `image_generation` entry is an inactive placeholder. Its `model`,
`endpoint`, and backend-specific `parameters` can be recorded in the catalog;
`enabled` must remain false until an image backend and adapter are integrated.
Image generation is not implemented or advertised as a working endpoint. The
inactive entry is preserved as metadata; backend-specific fields are not interpreted yet. The
Qwen chat models' `input: ["text", "image"]` describes image understanding;
it does not make them image generators. No image server is installed or started.

`make check` sends a normal and streaming request to **every enabled chat alias**,
reports failures per alias, and continues checking later models. This loads models
on demand and can take time. Use `make check MODEL=alias` to test just one. Native
Ollama model-list responses are validated before use, so malformed data becomes
an actionable error and does not skip later model-unload attempts.

The retired model-selection environment variables are ignored in current `.env`
files. Their values are read only from old saved sessions so shutdown can still
clean up a session created before catalog support. This project disables automatic
Python interpreter downloads; `make setup` requires an existing suitable Python.

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
| Model | Any enabled catalog alias, such as `local-coder` or `local-coder-bf16` |
| API key | Output of `make key NAME=laptop` |

In current VS Code: **Chat → Manage Language Models → Add Models → Custom Endpoint**.
Choose Chat Completions and configure the endpoint, key, and model. If the editor
asks for a complete request URL, use the base URL followed by `/chat/completions`.
Copilot Chat supplies the coding harness; this server supplies inference. Agent
mode depends on the chosen model's native tool-calling capabilities. Inline
autocomplete is a separate client feature.

Each client key is scoped to the configured model aliases. When `make key` reuses a
valid key after a model-list change, it updates that key's model permissions without
changing the raw key, preserving clients' secure-store entries. Keys not accessed
during startup can be refreshed with `make key NAME=client-name` after restarting.
Keep the LiteLLM master
key local; it is the administrative credential. `make key` displays a secret on
purpose. Use `make -s key NAME=laptop` if you need only the key without Make's
command echo. Local key copies live in `.runtime/keys/` with mode 0600. Revocation
removes both the gateway key and its local copy. The default `copilot` key is
created during startup for checks; other keys can be created as needed.

Run `make copilot-config` to write `.runtime/copilot-models.json` from the catalog.
Merge its `models` list into your existing VS Code Custom Endpoint provider, keeping
the existing secure API-key reference. The export contains no credentials and does
not edit VS Code. It includes only enabled chat models and uses their individual
context limits, display names, vision, tool-calling, and thinking capabilities.

For multiple Copilot choices, add one model entry per alias under the same Custom
Endpoint provider and reuse its existing API key. For the example above, use
`local-coder` / **Local Qwen MXFP8** and `local-coder-bf16` / **Local Qwen BF16**.
Select either in the Chat model picker; changing that choice requires no gateway restart.

`make key` and `make revoke` use the running session's configuration, including
its catalog aliases and gateway port. Editing configuration does not change their target;
apply those edits with `make restart` first if you want keys for the new configuration.

## Project layout

```text
Makefile                       Application entry point
pyproject.toml / uv.lock        uv project and locked dependencies
src/llm_inference/
    cli.py                     CLI dispatch
    bootstrap.py               First-use configuration and credential generation
    config.py                  Service settings, snapshots, and gateway/client rendering
    models.py                  Typed chat and inactive image-generation catalog entries
    errors.py                  User-facing application errors
    controller.py              Startup/shutdown orchestration
    runtime.py                 Compose, subprocesses, lifecycle locks, and readiness helpers
    ollama.py                  Model load/unload through the existing Ollama API
    gateway.py                 Client keys and gateway protocol checks
    transport.py               Shared HTTP requests and status checks
    storage.py                 Private atomic JSON writes
compose.yaml                   OrbStack container services
config/models.json             Model definitions, capabilities, and per-model settings
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

Each model's `timeout` is capped at Nginx's configured 900-second proxy timeout.
Nginx's read and send timeouts measure inactivity between operations, so this
does not enforce a 900-second wall-clock deadline for an entire streamed response.

References: [OrbStack networking](https://docs.orbstack.dev/docker/network),
[LiteLLM virtual keys](https://docs.litellm.ai/docs/proxy/virtual_keys),
[Ollama model lifetime](https://docs.ollama.com/faq#how-do-i-keep-a-model-loaded-in-memory-or-make-it-unload-immediately),
[VS Code custom endpoints](https://code.visualstudio.com/docs/agent-customization/language-models#add-a-custom-endpoint-model).
