.DEFAULT_GOAL := help
.PHONY: help setup quality lint types schemas test solve validate score demo verify-determinism docker-build docker-solve clean

INPUT ?= data/dataset.zip
OUTPUT ?= Submission.json
DATASET ?= agentic-bank-public

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

demo: ## Прогон на зафиксированном примере, без ключей API
	uv run halyk solve --input tests/fixtures/demo.zip --output artifacts/demo/Submission.json \
		--replay artifacts/example-run

verify-determinism: ## Два прогона из кэша, ответы обязаны совпасть
	uv run halyk solve --input $(INPUT) --output /tmp/first.json --replay artifacts/example-run
	uv run halyk solve --input $(INPUT) --output /tmp/second.json --replay artifacts/example-run
	diff /tmp/first.json /tmp/second.json && echo "совпадает"

docker-build:
	docker build -t halyk-agent .

docker-solve: ## Прогон в контейнере
	docker run --rm --env-file .env -v $(PWD)/data:/data -v $(PWD)/artifacts:/app/artifacts \
		halyk-agent solve --input /data/$(notdir $(INPUT)) --output /data/$(notdir $(OUTPUT))

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
