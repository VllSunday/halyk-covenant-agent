# Боевой runbook

Команды ниже даны для PowerShell из корня репозитория. Архив вручную распаковывать
не нужно: pipeline проверит пути внутри ZIP, найдёт шаблон и создаст отдельный
каталог прогона.

## До публикации датасета

1. Убедиться, что Docker Desktop запущен только если планируется контейнерный путь.
2. Проверить `.env`: `OPENAI_API_KEY`, `HALYK_TEAM`, `HALYK_CONTACT_EMAIL`.
3. Оставить `HALYK_MAX_CONCURRENCY=6`, `HALYK_MAX_LIVE_CALLS=80`,
   `HALYK_MAX_TOTAL_INPUT_TOKENS=2000000` и `HALYK_MAX_COST_USD=8`.
4. Выполнить локальные проверки:

```powershell
uv sync --all-groups --locked
make quality
make test
New-Item data, submission -ItemType Directory -Force
```

На счёте API нужен запас сверх измеренных $5,07 публичного прогона. Ограничитель
стоимости защищает pipeline, но недостаток средств у провайдера остановит его.

## Первый прогон

Скачать выданный архив без переименования файлов внутри и сохранить его как
`data/dataset.zip`. Затем:

```powershell
uv run halyk solve --input data/dataset.zip --output submission/attempt-1.json
```

Успех определяется одновременно тремя признаками:

- команда завершилась с кодом `0`;
- существует `submission/attempt-1.json`;
- последний `report.json` не содержит ошибок инвариантов.

Найти последний прогон и проверить его:

```powershell
$run = Get-ChildItem artifacts/runs -Directory |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

uv run halyk audit --run $run.FullName --all
$template = Get-ChildItem "$($run.FullName)/dataset" -Recurse `
  -Filter submission_template.json | Select-Object -First 1
uv run halyk validate --submission submission/attempt-1.json `
  --template $template.FullName
Get-FileHash submission/attempt-1.json -Algorithm SHA256
```

Если входом был уже распакованный каталог, шаблона внутри каталога прогона не
будет. В этом случае передать `--template` путь к `submission_template.json` из
исходного каталога.

После чистого audit первую попытку следует отправить сразу. У приватного набора
нет ключа, поэтому audit показывает полноту, риск и внутреннюю согласованность,
но не настоящий процент правильных ответов.

## Если прогон упал

Сначала открыть, не перезапуская модели:

```powershell
Get-Content "$($run.FullName)/report.json"
Get-Content "$($run.FullName)/metrics.json"
```

Исправлять нужно названный класс ошибки: неизвестный документ, открытое требование,
невалидную формулу, неоднозначный факт или бюджет. Не редактировать JSON ответа
вручную и не подставлять число из внешнего анализа.

После исправления запустить новую попытку обычной командой, без
`--no-cache-read`:

```powershell
uv run halyk solve --input data/dataset.zip --output submission/attempt-2.json
$candidate = Get-ChildItem artifacts/runs -Directory |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
uv run halyk compare --baseline $run.FullName --candidate $candidate.FullName
uv run halyk audit --run $candidate.FullName --all
```

Общий content-addressed cache повторно оплатит только запросы, вход которых
изменился. Вторую или третью попытку отправлять после чистого audit и отсутствия
регрессий в `compare`.

## Docker

Если локальное Python-окружение сломалось:

```powershell
docker compose build
docker compose run --rm halyk
```

Результат будет в `submission/attempt-1.json`. Docker и нативный запуск используют
одни и те же `.env`, `uv.lock`, модели, кэш и каталоги артефактов.

## Что сохранить после сдачи

Не публиковать приватный архив, общий model cache и содержимое каталога прогона.
Сохранить локально отправленный JSON, его SHA-256, `run_manifest.json`,
`cache_index.json`, `lineage.jsonl`, `metrics.json` и `report.json`. Организаторам
кэш передавать только по запросу.
