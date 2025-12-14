.PHONY: build up down logs shell clean

build:
	docker-compose build

buildnocache:
	docker-compose build --no-cache

up:
	docker-compose up -d

down:
	docker-compose down


# Check numpy and astropy versions inside the container
numpy_astropy_version:
	docker run --rm --entrypoint python fitsreader-app -c "import numpy, astropy; print(numpy.__version__, astropy.__version__)"


# Check Python version inside the container
python_version:
	docker run --rm --entrypoint python fitsreader-app -V


logs:
	docker-compose logs -f

shell:
	docker-compose exec app /bin/bash

clean:
	docker-compose down -v
	docker system prune -f
