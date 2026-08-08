"""Сквозной прогон и всё, что ему нужно, чтобы дойти от датасета до ответа."""

from halyk.pipeline.engines import Engines, build_engines
from halyk.pipeline.solve import PipelineError, SolveResult, solve

__all__ = ["Engines", "PipelineError", "SolveResult", "build_engines", "solve"]
