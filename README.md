# Local LLMs Stack

## Install

Will create Ollama, Web Open UI, and ComfyUI as a docker container always running in the background.

- http://localhost:8111 - Web Open UI
- http://localhost:8112 - ComfyUI
- http://host.docker.internal:8112 - ComfyUI Docker Containers
- http://localhost:11434 - Ollama
- http://host.docker.internal:11434 - Ollama Docker Containers

```bash
make build
```

## Creating a Open Hands Session (Open Devin)

```bash
make run_openhands
```

# create a new model from file

```bash
make ollama_model name=<model_name> file=<model_file>
```

