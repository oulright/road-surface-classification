.PHONY: setup setup-dvc setup-kaggle train-audio train-video test lint format clean dvc-init dvc-push dvc-pull dvc-status kaggle-push colab-notebook notebook notebook-lab

# ── Установка ──
setup:
	pip install -e ".[dev]"
	pre-commit install

setup-dvc:
	pip install -e ".[dev,dvc]"

setup-kaggle:
	pip install -e ".[kaggle]"

# ── DVC ──
dvc-init:
	dvc init
	dvc remote add -d storage s3://$(BUCKET)/dvc-storage
	dvc remote modify storage endpointurl https://storage.yandexcloud.net

dvc-push:
	dvc push

dvc-pull:
	dvc pull

dvc-status:
	dvc status

# ── Обучение ──
train-audio:
	python scripts/train.py --config-path ../configs/audio/models \
	                        --config-name resnet18_mel

train-video:
	python scripts/train.py --config-path ../configs/video/models \
	                        --config-name efficientnet_b2

# ── Оценка ──
evaluate:
	python scripts/evaluate.py --checkpoint $(CKPT) --config $(CONFIG)

# ── Тесты ──
test:
	pytest tests/ -v --tb=short

test-core:
	pytest tests/core/ -v

test-audio:
	pytest tests/audio/ -v

test-video:
	pytest tests/video/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html

# ── Код ──
lint:
	ruff check src/ scripts/ tests/

format:
	ruff format src/ scripts/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache htmlcov .ruff_cache

# ── Kaggle ──
kaggle-push:
	python scripts/sync_kaggle_kernel.py $(NOTEBOOK) \
		$(if $(TITLE),--title "$(TITLE)",) \
		$(if $(WAIT),--wait,) \
		$(if $(GPU),,$(if $(filter $(GPU),0 false),--no-gpu,)) \
		$(if $(INTERNET),,$(if $(filter $(INTERNET),0 false),--no-internet,))

# ── Google Colab ──
colab-notebook:
	@echo ""
	@echo "Откройте ноутбук в Google Colab:"
	@echo "  1. Перейдите в https://colab.research.google.com"
	@echo "  2. File → Open notebook → GitHub"
	@echo "  3. Вставьте URL репозитория"
	@echo "  4. Выберите notebooks/colab/train.ipynb"
	@echo ""
	@echo "Или загрузите файл вручную:"
	@echo "  notebooks/colab/train.ipynb"
	@echo ""

# ── Notebooks ──
notebook:
	jupyter notebook notebooks/

notebook-lab:
	jupyter lab notebooks/
