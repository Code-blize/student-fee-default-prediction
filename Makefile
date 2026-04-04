install:
	pip install -r requirements.txt
	pip install pytest httpx

test:
	pytest tests -q

run-api:
	uvicorn src.api.main:app --reload

docker-build:
	docker build -t fee-defaulter-api .

docker-run:
	docker run -p 8000:8000 fee-defaulter-api
