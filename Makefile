.PHONY: setup data train export serve test monitor anomaly docker-build k8s-apply clean help

help:
	@echo "=========================================================================="
	@echo "  NEU-DET Steel Surface Defect Inspection Service (uv Native)"
	@echo "=========================================================================="
	@echo "make setup        - Install dependencies and sync environment via uv sync"
	@echo "make data         - Download raw dataset, convert XML to YOLO & split"
	@echo "make augment      - Generate lighting-shift factory evaluation benchmark"
	@echo "make train        - Run MLflow YOLOv8 experiment tracking suite"
	@echo "make export       - Export best model to ONNX FP32 and INT8, benchmark CPU"
	@echo "make monitor      - Compute baseline statistics and run KS-test drift demo"
	@echo "make anomaly      - Fit PatchCore unsupervised anomaly detection baseline"
	@echo "make serve        - Start production FastAPI ONNX inference server via uv"
	@echo "make test         - Run full pytest test suite via uv run pytest"
	@echo "make docker-build - Build production container image with uv"
	@echo "make k8s-apply    - Deploy to Kubernetes cluster (Deployment + Service)"
	@echo "=========================================================================="

setup:
	uv sync --all-groups

data:
	uv run python data/convert_and_split.py

augment:
	uv run python src/augment.py

train:
	uv run python src/train.py

export:
	uv run python src/export.py --weights models/champion_best.pt

monitor:
	uv run python src/monitor/drift.py --build-baseline
	uv run python src/monitor/drift.py --demo

anomaly:
	uv run python src/anomaly/patchcore.py --eval

serve:
	uv run uvicorn src.serve.app:app --host 0.0.0.0 --port 8000 --reload

test:
	uv run pytest -v tests/

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
