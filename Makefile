DATA_DIR = $(pwd)
build:
	echo "Building... in $(DATA_DIR)"
	./build.sh
ollama_image_pull:
	docker exec -it ollama ollama pull $(image)
ollama_image_list:
	docker exec -it ollama ollama list
ollama_model:
	docker exec -it ollama ollama create $(name) -f $(file)
ollama_model_delete:
	docker exec -it ollama ollama rm $(name)
run_openhands:
	./run_openhands.sh
run_devika:
	./run_devika.sh
