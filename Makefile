PY ?= python
BITEXT ?= data/bitext.csv

.PHONY: install sample train test run siebel dashboard certs up down logs load deploy-sm test-sm delete-sm

install:          ## Dependencias de desarrollo
	$(PY) -m pip install -r requirements-dev.txt

sample:           ## CSV sintético con el esquema Bitext (solo para pruebas)
	$(PY) -m training.bitext --out data/bitext_sample.csv

train:            ## Entrena NLU + índice RAG (usa BITEXT=... para el dataset real de Kaggle)
	@if [ -f $(BITEXT) ]; then $(PY) -m training.train --bitext $(BITEXT) --out models; \
	else echo "No existe $(BITEXT); usando muestra sintética"; $(MAKE) sample; $(PY) -m training.train --bitext data/bitext_sample.csv --out models; fi

test:             ## Pruebas unitarias y de integración
	$(PY) -m pytest -v

run:              ## Backend local sin Docker (Redis/colas/BD en memoria, Siebel simulado en proceso)
	SIEBEL_BASE_URL=inprocess $(PY) -m uvicorn app.main:app --reload --port 8000

dashboard:
	API_URL=http://localhost:8000 streamlit run dashboard/app.py

certs:
	./nginx/gen-cert.sh $(or $(HOST),localhost)

up: certs         ## Stack completo con Docker
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f backend siebel-worker

load:             ## Prueba de carga: make load HOST=https://IP USERS=300
	mkdir -p loadtest/results
	locust -f loadtest/locustfile.py --host $(HOST) --headless -u $(or $(USERS),300) -r 30 -t 5m \
	  --csv loadtest/results/carga --html loadtest/results/carga.html
	$(PY) loadtest/summarize.py loadtest/results/carga_stats.csv

deploy-sm:        ## Training Job + endpoint SageMaker: make deploy-sm BUCKET=telecombot-xxx
	$(PY) -m aws.sagemaker_deploy deploy --bucket $(BUCKET)

endpoint-sm:      ## Solo (re)crea el endpoint usando el último Training Job completado
	$(PY) -m aws.sagemaker_deploy deploy --bucket $(BUCKET) --reuse-training

test-sm:
	$(PY) -m aws.sagemaker_deploy test

delete-sm:        ## ¡Eliminar el endpoint al terminar para no gastar créditos!
	$(PY) -m aws.sagemaker_deploy delete
