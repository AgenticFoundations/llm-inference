# LLM inference project context

General user-control and installation preferences are recorded globally in
`~/.codex/AGENTS.md`. This file contains only the project's design and operation context.

## Ollama design

- Use the user's existing Ollama installation and running server. Do not install another
  Ollama, run an Ollama container, launch another `ollama serve`, change login behavior,
  or stop the Ollama application/server.
- The intended LLM lifecycle is model control through the existing Ollama API: preload the
  explicitly selected catalog model or unload configured models, leaving the server and model files intact.
  Unloading must target only configured models; explain that other clients using the
  same models share their loaded state. All active chat aliases are selected per request.
- Model definitions and per-model settings belong in `config/models.json`, not model-selection
  environment variables. Image-generation entries remain inactive until a backend is integrated.
- During shutdown, stop only this project's containers and unload its configured models.
  Leave unrelated applications, models, containers, and networks alone.
- Do not automatically download models, run inference, change VS Code configuration, or
  store credentials in another application during discussion or cleanup.

## Project scope

This project owns the local LLM gateway, database, Nginx API proxy, and model load/unload
requests. Keep its configuration self-contained. Network connectivity between its
containers is managed by this project's Compose configuration.
Do not start, stop, or restart running services as part of a code/configuration cleanup
unless the user explicitly asks to apply the changes to the running stack.
