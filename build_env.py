"""
Script to install
"""

import argparse
import json
import multiprocessing
import os
import subprocess
import time


class Installer:
    def __init__(self, args):
        self.log_file = "install.log"
        self.base_dir = os.getcwd()
        self.workspace_dir = args.workspace_dir
        self.postgres_user = args.postgres_user
        self.postgres_password = args.postgres_password
        self.postgres_db = args.postgres_db
        self.n8n_encryption_key = args.n8n_encryption_key
        self.n8n_user_management_jwt_secret = args.n8n_user_management_jwt_secret
        self.ollama_docker = args.ollama_docker == "yes"
        self.choices = args.services or [
            "OpenWebUIGPU",
            "OpenWebUI Pipelines",
            "ComfyUI",
            "Postgres",
            "pgAdmin",
            "n8n",
            "Qdrant",
            "ComfyUIVP",
        ]
        self.models = args.models or ["llama3.1"]

        # check if self.base_dir/extra_models/sd_xl_base_1.0_0.9vae.safetensors exists
        # https://github.com/thecooltechguy/ComfyUI-Stable-Video-Diffusion
        self.video_models = {
            "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/resolve/main/svd_image_decoder.safetensors?download=true": f"{self.base_dir}/extra_models/svd_image_decoder.safetensors",
            "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/resolve/main/svd.safetensors?download=true": f"{self.base_dir}/extra_models/svd.safetensors",
            "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt.safetensors?download=true": f"{self.base_dir}/extra_models/svd_xt.safetensors",
            "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt_image_decoder.safetensors?download=true": f"{self.base_dir}/extra_models/svd_xt_image_decoder.safetensors",
        }

    def log(self, message):
        with open(self.log_file, "a") as log_file:
            log_file.write(f"{time.strftime('%Y-%m-%d %T')} - {message}\n")
        print(message)

    def install_dialog(self):
        if subprocess.call("command -v dialog", shell=True) != 0:
            self.log("Installing dialog...")
            if os.name == "posix":
                subprocess.call(
                    "sudo apt-get update && sudo apt-get install -y dialog", shell=True
                )
            elif os.name == "darwin":
                subprocess.call("brew install dialog", shell=True)
            else:
                self.log("Unsupported OS for automatic dialog installation.")
                exit(1)

    def check_service(self, service_name, docker_name):
        if subprocess.call(f"docker ps -q -f name={docker_name}", shell=True) == 0:
            if (
                input(
                    f"{service_name} is already running. Do you want to rebuild the Docker image? (y/n): "
                ).lower()
                == "y"
            ):
                self.log(f"Rebuilding {service_name}...")
                subprocess.call(
                    f"docker stop {docker_name} && docker rm {docker_name}", shell=True
                )
            else:
                self.log(f"Skipping rebuild of {service_name}.")
                return False
        return True

    def pull_models(self):
        for model in self.models:
            if subprocess.call("command -v ollama", shell=True) == 0:
                subprocess.call(f"ollama pull {model}", shell=True)
            else:
                subprocess.call(f"docker exec ollama ollama pull {model}", shell=True)

    def create_servers_json(self):
        servers = {
            "Servers": [
                {
                    "name": "localhost",
                    "host": "host.docker.internal",
                    "port": 5432,
                    "user": self.postgres_user,
                    "database": self.postgres_db,
                    "PassFile": "/pgadmin4/pgpass",
                }
            ]
        }
        with open("servers.json", "w") as f:
            json.dump(servers, f, indent=4)

        pgpass_string = f"host.docker.internal:5432:{self.postgres_db}:{self.postgres_user}:{self.postgres_password}"

        with open("servers.pgpass", "w") as f:
            f.write(pgpass_string)

    def install_services(self):
        workspace_volume_docker_args = f"-v {self.workspace_dir}:/workspace"
        openwebui_args = (
            "-e PORT=8111 --network=host -e OLLAMA_API_BASE_URL=http://127.0.0.1:11434"
        )
        n8n_args = (
            "-e PORT=5678 --network=host -e OLLAMA_API_BASE_URL=http://127.0.0.1:11434"
        )
        if self.ollama_docker:
            openwebui_args = "-p 8111:8080 --add-host=host.docker.internal:host-gateway"
            n8n_args = "-p 5678:5678 --add-host=host.docker.internal:host-gateway"
        for choice in self.choices:
            if choice == "OpenWebUIGPU" and self.check_service(
                "OpenWebUI (GPU)", "open-webui"
            ):
                self.log("Installing OpenWebUI (GPU)...")
                command = f"""
docker run -d 
{openwebui_args} 
{workspace_volume_docker_args} 
--gpus all 
--name open-webui 
-v {self.base_dir}/OpenWebUIData:/app/backend/data 
-e GLOBAL_LOG_LEVEL='DEBUG' 
--restart always 
ghcr.io/open-webui/open-webui:cuda
                """
                subprocess.call(
                    " ".join(command.split("\n")),
                    shell=True,
                )
            elif choice == "OpenWebUI" and self.check_service(
                "OpenWebUI", "open-webui"
            ):
                command = f"""
docker run -d 
{openwebui_args} 
{workspace_volume_docker_args} 
-e GLOBAL_LOG_LEVEL='DEBUG' 
-v {self.base_dir}/OpenWebUIData:/app/backend/data 
--restart always 
--name open-webui 
ghcr.io/open-webui/open-webui:main
                """
                self.log("Installing OpenWebUI...")
                subprocess.call(
                    " ".join(command.split("\n")),
                    shell=True,
                )
            elif choice == "OpenWebUI Pipelines" and self.check_service(
                "OpenWebUI Pipelines", "open-webui-pipelines"
            ):
                self.log("Installing OpenWebUI Pipelines...")
                command = f"""
docker run -d 
{'-p 9099:9099 --add-host=host.docker.internal:host-gateway' if self.ollama_docker else '-e PORT=9099 --network=host'} 
-v {self.base_dir}/OpenWebUIPipelinesData:/app/pipelines 
--restart always  
--name open-webui-pipelines
ghcr.io/open-webui/pipelines:main
"""
                subprocess.call(
                    " ".join(command.split("\n")),
                    shell=True,
                )
            elif choice == "ComfyUI" and self.check_service("ComfyUI", "comfyui"):
                self.log("Installing ComfyUI...")
                # -v {self.base_dir}/ComfyUIData:/home/runner
                command = f"""
docker run -d 
--gpus all 
{'-p 8188:8188 --add-host host.docker.internal:host-gateway' if self.ollama_docker else '-e PORT=9099 --network=host'} 
-e CLI_ARGS='' 
--restart always 
--name comfyui 
yanwk/comfyui-boot:cu121
"""
                subprocess.call(
                    " ".join(command.split("\n")),
                    shell=True,
                )
            elif choice == "ComfyUIVP":
                self.log("Installing ComfyUI (Video Package)...")
                #                 command = f"""
                # docker exec comfyui bash -c
                # 'cd /home/runner/ComfyUI/custom_nodes/ &&
                # zypper install -y python3 wget &&
                # rm -Rf ComfyUI-Stable-Video-Diffusion &&
                # mkdir /home/runner/ComfyUI/models/svd &&
                # git clone https://github.com/thecooltechguy/ComfyUI-Stable-Video-Diffusion &&
                # cd ComfyUI-Stable-Video-Diffusion/ &&
                # python3 install.py'
                #                 """
                # subprocess.call(
                #     " ".join(command.split("\n")),
                #     shell=True,
                # )
                with multiprocessing.Pool() as pool:
                    pool.starmap(
                        self.download_and_copy_model, self.video_models.items()
                    )

            elif choice == "Postgres" and self.check_service("Postgres", "postgresllm"):
                self.log("Installing Postgres...")
                command = f"""
docker run -d 
--restart=unless-stopped 
{'-p 5432:5432 --add-host host.docker.internal:host-gateway' if self.ollama_docker else '-e PORT=5432 --network=host'} 
-e POSTGRES_USER={self.postgres_user} 
-e POSTGRES_PASSWORD={self.postgres_password} 
-e POSTGRES_DB={self.postgres_db} 
-v {self.base_dir}/PostgreSQLData:/var/lib/postgresql/data 
--health-cmd='pg_isready -h localhost -U {self.postgres_user} -d {self.postgres_db}' 
--health-interval=5s 
--health-timeout=5s 
--health-retries=10
--name postgresllm 
postgres:16-alpine
"""
                subprocess.call(
                    " ".join(command.split("\n")),
                    shell=True,
                )
            elif choice == "pgAdmin" and self.check_service("pgAdmin Web", "pgadmin"):
                self.log("Installing pgAdmin Web...")
                self.create_servers_json()
                command = f"""
docker run -d 
{'-p 5050:80 --add-host host.docker.internal:host-gateway' if self.ollama_docker else '-e PORT=5432 --network=host -e PGADMIN_LISTEN_PORT=5050'} 
-v {self.base_dir}/servers.json:/pgadmin4/servers.json 
-e PGADMIN_DEFAULT_EMAIL='root@root.io' 
-e PGADMIN_DEFAULT_PASSWORD={self.postgres_password} 
--restart always 
--name pgadmin 
dpage/pgadmin4
"""
                subprocess.call(
                    " ".join(command.split("\n")),
                    shell=True,
                )

                subprocess.call(
                    f"docker container cp {self.base_dir}/servers.json pgadmin:/pgadmin4/servers.json",
                    shell=True,
                )

                subprocess.call(
                    f"docker container cp {self.base_dir}/servers.pgpass pgadmin:/pgadmin4/pgpass",
                    shell=True,
                )
            elif choice == "n8n" and self.check_service("n8n", "n8n"):
                self.log("Installing n8n...")
                command = f"""
docker run -d 
{workspace_volume_docker_args}
{n8n_args} 
--restart always  
--name n8n 
docker.n8n.io/n8nio/n8n
"""
                subprocess.call(
                    " ".join(command.split("\n")),
                    shell=True,
                )
            elif choice == "Qdrant" and self.check_service("Qdrant", "qdrant"):
                self.log("Installing Qdrant...")
                command = f"""
docker run -d 
{'-p 6333:6333 -p 6334:6334 --add-host host.docker.internal:host-gateway' if self.ollama_docker else '-e PORT=6334 -e PORT=6333 --network=host'} 
--restart always 
-v {self.base_dir}/qdrant-data:/qdrant/storage:z 
--name qdrant 
qdrant/qdrant
"""
                subprocess.call(
                    " ".join(command.split("\n")),
                    shell=True,
                )
            elif choice == "Browserless" and self.check_service(
                "Browserless", "browserless"
            ):
                self.log("Installing Browserless...")
                command = f"""
docker run -d
-p 8999:3000
-e "CONCURRENT=10" 
-e "CORS=true" 
-e "CORS_MAX_AGE=300" 
-e "TOKEN=6R0W53R135510" 
-v {self.base_dir}/browserless:/root
--name browserless 
ghcr.io/browserless/chromium
"""
                subprocess.call(
                    " ".join(command.split("\n")),
                    shell=True,
                )

    def download_and_copy_model(self, model, destination):
        if not os.path.exists(destination):
            subprocess.call(f"wget {model} -O {destination}", shell=True)
        subprocess.call(
            f"docker container cp {destination} comfyui:/home/runner/ComfyUI/models/svd/",
            shell=True,
        )

    def print_urls(self):
        self.log("OpenWebUI: http://localhost:8111")
        self.log("OpenWebUI Pipelines: http://localhost:9099")
        self.log("ComfyUI: http://localhost:8188")
        self.log("pgAdmin Web: http://localhost:5050")
        self.log("n8n: http://localhost:5678")
        self.log("Qdrant: http://localhost:6333")

    def run(self):
        self.install_dialog()
        self.pull_models()
        self.install_services()
        self.print_urls()
        self.log("Installation completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install various services.")
    parser.add_argument(
        "--ollama-docker",
        default="no",
        nargs=1,
        choices=["yes", "no"],
        help="Set if Ollama is running in Docker",
    )
    parser.add_argument(
        "--workspace-dir",
        default=os.path.expanduser("~/workspace"),
        help="Workspace directory",
    )
    parser.add_argument("--postgres-user", default="root", help="Postgres user")
    parser.add_argument(
        "--postgres-password", default="password", help="Postgres password"
    )
    parser.add_argument("--postgres-db", default="n8n", help="Postgres database")
    parser.add_argument(
        "--n8n-encryption-key", default="super-secret-key", help="n8n encryption key"
    )
    parser.add_argument(
        "--n8n-user-management-jwt-secret",
        default="even-more-secret",
        help="n8n user management JWT secret",
    )
    parser.add_argument(
        "--services",
        nargs="+",
        choices=[
            "OpenWebUIGPU",
            "OpenWebUI",
            "OpenWebUI Pipelines",
            "ComfyUI",
            "ComfyUIVP",
            "Postgres",
            "pgAdmin",
            "n8n",
            "Qdrant",
            "Browserless",
        ],
        help="Services to install",
    )
    parser.add_argument("--models", nargs="+", help="Ollama models to pull")

    args = parser.parse_args()
    installer = Installer(args)
    installer.run()
