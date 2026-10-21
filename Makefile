.PHONY: setup data train export serve test monitor anomaly docker-build k8s-apply clean help

VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

help:
	@echo "=========================================================================="
	@echo "  NEU-DET Steel Surface Defect Inspection Service - Makefile"
	@echo "=========================================================================="
	@echo "make setup        - Create virtual environment and install dependencies"
	@echo "make data         - Download raw dataset, convert XML to YOLO & split"
	@echo "make augment      - Generate lighting-shift factory evaluation benchmark"
	@echo "make train        - Run MLflow YOLOv8 experiment tracking suite"
	@echo "make export       - Export best model to ONNX FP32 and INT8, benchmark CPU"
	@echo "make monitor      - Compute baseline statistics and run KS-test drift demo"
	@echo "make anomaly      - Fit PatchCore unsupervised anomaly detection baseline"
	@echo "make serve        - Start production FastAPI ONNX inference server"
	@echo "make test         - Run full pytest test suite"
	@echo "make docker-build - Build production container image"
	@echo "make k8s-apply    - Deploy to Kubernetes cluster (Deployment + Service)"
	@echo "=========================================================================="

setup:
	uv venv --python /usr/local/bin/python3 $(VENV)
	uv pip install --python $(PYTHON) -r requirements.txt

data:
	$(PYTHON) data/convert_and_split.py

augment:
	$(PYTHON) src/augment.py

train:
	$(PYTHON) src/train.py

export:
	$(PYTHON) src/export.py --weights models/champion_best.pt

monitor:
	$(PYTHON) src/monitor/drift.py --build-baseline
	$(PYTHON) src/monitor/drift.py --demo

anomaly:
	$(PYTHON) src/anomaly/patchcore.py --eval

serve:
	uvicorn src.serve.app:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest -v tests/

docker-build:
	docker build -t defect-inspection-service:latest .

k8s-apply:
	kubectl apply -f k8s/configmap.yaml
	kubectl apply -f k8s/deployment.yaml
	kubectl apply -f k8s/service.yaml
	kubectl apply -f k8s/hpa.yaml

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
