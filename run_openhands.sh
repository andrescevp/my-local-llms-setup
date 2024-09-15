# OpenHands (OpenDevin) - container per session
docker run -it \
  --pull=always \
  --add-host host.docker.internal:host-gateway \
  -e SANDBOX_RUNTIME_CONTAINER_IMAGE=ghcr.io/all-hands-ai/runtime:0.9-nikolaik \
  -e SANDBOX_USER_ID=$(id -u) \
  -e WORKSPACE_MOUNT_PATH=$DATA_DIR/OpenHandsData \
  -e LLM_API_KEY="ollama" \
  -e LLM_BASE_URL="http://host.docker.internal:11434" \
  -e LLM_OLLAMA_BASE_URL="http://host.docker.internal:11434" \
  -v $DATA_DIR/OpenHandsData:/opt/workspace_base \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -p 8112:3000 \
  --add-host host.docker.internal:host-gateway \
#  --name openhands-app-$(date +%Y%m%d%H%M%S) \
  --name openhands-app \
  ghcr.io/all-hands-ai/openhands:0.9