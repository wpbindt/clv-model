DOCKER_DIR := docker
start-jupyter :
	docker-compose -f '${DOCKER_DIR}/docker-compose.yml' up -d jupyter
stop-jupyter :
	docker-compose -f '${DOCKER_DIR}/docker-compose.yml' stop jupyter
compile-stan-models :
	docker-compose -f docker/docker-compose.yml run --rm cli python3 scripts/compile_stan_models.py

