from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from halyk import __version__
from halyk.config import Settings
from halyk.hashing import sha256_file, sha256_path
from halyk.knowledge.errata import ErrataRegistry
from halyk.llm.cache import CachePolicy
from halyk.models.manifest import RunManifest, RunMode, RunStatus

MANIFEST_NAME = "run_manifest.json"
METRICS_NAME = "metrics.json"
LINEAGE_NAME = "lineage.jsonl"
REPORT_NAME = "report.json"
SUBMISSION_NAME = "submission.json"
CACHE_DIR_NAME = "model_cache"


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


@dataclass(frozen=True, slots=True)
class RunContext:
    """Каталог прогона и всё, что о нём известно на старте."""

    run_id: str
    mode: RunMode
    root: Path
    settings: Settings
    started_at: datetime
    cache_read: bool = False

    @property
    def cache_dir(self) -> Path:
        return self.root / CACHE_DIR_NAME

    @property
    def lineage_path(self) -> Path:
        return self.root / LINEAGE_NAME

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_NAME

    @property
    def metrics_path(self) -> Path:
        return self.root / METRICS_NAME

    @property
    def report_path(self) -> Path:
        return self.root / REPORT_NAME

    @property
    def submission_path(self) -> Path:
        return self.root / SUBMISSION_NAME

    @property
    def cache_policy(self) -> CachePolicy:
        """Как этот прогон обращается с кэшем.

        Новый прогон кэш только пишет: молчаливое чтение записей от прошлого
        датасета дало бы ответы не про те документы. Чтение включается явно — флагом,
        продолжением прогона или его повторением.
        """
        if self.settings.offline:
            return CachePolicy.OFFLINE
        if self.mode is RunMode.REPLAY:
            return CachePolicy.REPLAY
        if self.mode is RunMode.RESUME or self.cache_read:
            return CachePolicy.READ_WRITE
        return CachePolicy.WRITE_ONLY

    @classmethod
    def create(
        cls,
        settings: Settings,
        input_path: Path,
        mode: RunMode = RunMode.SOLVE,
        run_id: str | None = None,
        *,
        cache_read: bool = False,
    ) -> RunContext:
        started = datetime.now(UTC)
        if run_id is None:
            stamp = started.strftime("%Y%m%d-%H%M%S")
            run_id = f"{stamp}-{sha256_path(input_path)[:8]}"
        root = settings.artifacts_dir / "runs" / run_id
        (root / CACHE_DIR_NAME).mkdir(parents=True, exist_ok=True)
        return cls(
            run_id=run_id,
            mode=mode,
            root=root,
            settings=settings,
            started_at=started,
            cache_read=cache_read,
        )

    def build_manifest(
        self,
        input_path: Path,
        replayed_from: str | None = None,
        *,
        template_path: Path | None = None,
        prompts: dict[str, str] | None = None,
    ) -> RunManifest:
        lock = Path("uv.lock")
        return RunManifest(
            run_id=self.run_id,
            mode=self.mode,
            started_at=self.started_at,
            input_sha256=sha256_path(input_path),
            input_name=input_path.name,
            template_sha256=sha256_file(template_path) if template_path else None,
            prompts_sha256=prompts or {},
            git_commit=_git("rev-parse", "HEAD"),
            git_dirty=bool(_git("status", "--porcelain")),
            python_version=platform.python_version(),
            dependency_lock_sha256=sha256_file(lock) if lock.exists() else "",
            halyk_version=__version__,
            models=self.settings.model_versions(),
            max_concurrency=self.settings.max_concurrency,
            errata_sha256=ErrataRegistry.load().digest,
            replayed_from=replayed_from,
        )

    def write_manifest(self, manifest: RunManifest) -> None:
        self.manifest_path.write_text(
            manifest.model_dump_json(indent=2, exclude_none=False),
            encoding="utf-8",
        )

    def finish(self, manifest: RunManifest, status: RunStatus, **updates: object) -> RunManifest:
        final = manifest.model_copy(
            update={
                "finished_at": datetime.now(UTC),
                "status": status,
                **updates,
            }
        )
        self.write_manifest(final)
        return final
