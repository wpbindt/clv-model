DOCKER_COMPOSE := docker-compose -f './docker/docker-compose.yml' 
CLI := $(DOCKER_COMPOSE) run --rm cli
clean :
	rm clv_model/stan_models/*.pkl
start-jupyter :
	$(DOCKER_COMPOSE) up -d jupyter
stop-jupyter :
	$(DOCKER_COMPOSE) stop jupyter
compile-stan-models :
	$(CLI) python3 scripts/compile_stan_models.py
run-tests :
	$(CLI) python3 -m unittest discover tests || true
