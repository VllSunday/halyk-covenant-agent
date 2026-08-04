# Запасной путь запуска. Основной — uv run, см. README.
FROM python:3.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9.10 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Зависимости отдельным слоем: они меняются реже кода.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

COPY src ./src
COPY schemas ./schemas
RUN uv sync --locked --no-dev


FROM python:3.12-slim-bookworm

WORKDIR /app
COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH"

RUN useradd --create-home --uid 1000 halyk && chown -R halyk:halyk /app
USER halyk

ENTRYPOINT ["halyk"]
CMD ["--help"]
