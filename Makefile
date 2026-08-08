.DEFAULT_GOAL := help
.PHONY: help setup quality lint types schemas test solve validate score verify-determinism docker-build docker-solve clean

INPUT ?= data/dataset.zip
OUTPUT ?= Submission.json
DATASET ?= agentic-bank-public
RUN ?=

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

setup: ## Поставить зависимости из lock-файла
	uv sync --all-groups --locked

lint: ## ruff
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

types: ## mypy
	uv run mypy src

quality: lint types ## Всё, что должно быть зелёным перед коммитом
	uv run python scripts/export_schemas.py --check

schemas: ## Перегенерировать JSON Schema из моделей
	uv run python scripts/export_schemas.py

test: ## pytest без тестов, требующих сети
	uv run pytest -m "not integration"

solve: ## Полный прогон: make solve INPUT=data/dataset.zip
	uv run halyk solve --input $(INPUT) --output $(OUTPUT)

validate: ## Проверить ответ по схеме и по составу ячеек шаблона
	uv run halyk validate --submission $(OUTPUT) --template $(DATASET)/submission_template.json

score: ## Оценить ответ по ключу открытого датасета; только для разработки
	uv run halyk score --submission $(OUTPUT) --ground-truth $(DATASET)/ground_truth.json

verify-determinism: ## Побайтно повторить RUN дважды: make verify-determinism RUN=...
	@test -n "$(RUN)" || (echo "задайте RUN=artifacts/runs/<run_id>" && exit 2)
	uv run halyk solve --input $(INPUT) --output artifacts/replay-first.json --replay $(RUN)
	uv run halyk solve --input $(INPUT) --output artifacts/replay-second.json --replay $(RUN)
	diff artifacts/replay-first.json artifacts/replay-second.json

docker-build:
	docker build -t halyk-agent .

docker-solve: ## Прогон в контейнере
	docker run --rm --env-file .env -v $(PWD)/data:/data -v $(PWD)/artifacts:/app/artifacts \
		halyk-agent solve --input /data/$(notdir $(INPUT)) --output /data/$(notdir $(OUTPUT))

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
