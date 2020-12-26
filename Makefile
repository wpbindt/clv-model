DOCKER_DIR := docker
start-jupyter :
	docker-compose -f '${DOCKER_DIR}/docker-compose.yml' up -d jupyter
stop-jupyter :
	docker-compose -f '${DOCKER_DIR}/docker-compose.yml' stop jupyter

