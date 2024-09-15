# Requirements:
# - Docker
# - Nvidia Toolkit
# Ollama Install
# https://ollama.com/blog/ollama-is-now-available-as-an-official-docker-image
# install nvidia toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation
export DATA_DIR=$(pwd)

# Ollama CPU
# docker run -d \
#  --add-host host.docker.internal:host-gateway \
#  -v $DATA_DIR/Ollama:/root/.ollama \
#  --restart always \
#  -p 11434:11434 \
#  --name ollama \
#  ollama/ollama

echo "Building LLM Local Stack... in $DATA_DIR"
# Ollama Nvidia GPU
docker run \
  -d \
  --gpus=all \
  --add-host host.docker.internal:host-gateway \
  -v $DATA_DIR/OllamaData:/root/.ollama \
  -p 11434:11434 \
  --restart always \
  --name ollama \
  ollama/ollama

# web ui for ollama
docker run -d \
  -p 8111:8080 \
  --add-host=host.docker.internal:host-gateway \
  -v $DATA_DIR/OpenWebUIData:/app/backend/data \
  --name open-webui \
  --restart always \
  --add-host host.docker.internal:host-gateway \
  ghcr.io/open-webui/open-webui:0.3

# ConfyUI
docker run -d \
  --name comfyui \
  --gpus all \
  -p 8188:8188 \
  -v $DATA_DIR/ComfyUIData:/home/runner \
  -e CLI_ARGS="" \
  --add-host host.docker.internal:host-gateway \
  --restart always \
  yanwk/comfyui-boot:cu121

# check if devika-official-repo exists
if [ ! -d "devika-official-repo" ]; then
  git clone https://github.com/stitionai/devika ./devika-official-repo
fi

echo "Pulling nomic-embed-text"
docker exec -ti ollama ollama pull nomic-embed-text
echo "Pulling phi3"
docker exec -ti ollama ollama pull phi3
echo "Pulling ollama3.1"
docker exec -ti ollama ollama pull ollama3.1
echo "Pulling llava"
docker exec -ti ollama ollama pull llava
echo "Pulling codellama"
docker exec -ti ollama ollama pull codellama
echo "Pulling gemma2"
docker exec -ti ollama ollama pull gemma2
echo "Pulling mistral-nemo"
docker exec -ti ollama ollama pull mistral-nemo
echo "Pulling qwen2"
docker exec -ti ollama ollama pull qwen2
echo "Pulling deepseek-coder-v2"
docker exec -ti ollama ollama pull deepseek-coder-v2
echo "Pulling mistral"
docker exec -ti ollama ollama pull mistral
echo "Pulling mixtral"
docker exec -ti ollama ollama pull mixtral


