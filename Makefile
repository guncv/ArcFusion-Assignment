dc = docker compose -f docker-compose.yaml

.PHONY: run down build clean logs restart ps migrate-up migrate-down rebuild

info:
	$(dc) ps

run:
	$(dc) up

down:
	$(dc) down

build:
	$(dc) build

clean:
	$(dc) down --rmi all --volumes --remove-orphans

logs:
	$(dc) logs -f

restart:
	$(dc) restart

ps:
	$(dc) ps

rebuild: clean build run
