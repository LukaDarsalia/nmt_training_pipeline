"""Evaluation metric registry and metric implementations."""

from __future__ import annotations

from typing import Callable, Dict, List

import evaluate


class MetricRegistry:
    """A registry for mapping metric names to callable evaluation functions."""

    def __init__(self) -> None:
        self._metrics: Dict[str, Callable[[List[str], List[str]], Dict[str, float]]] = {}
        self._descriptions: Dict[str, str] = {}

    def register(self, name: str, description: str = "") -> Callable[[Callable[..., Dict[str, float]]], Callable[..., Dict[str, float]]]:
        """Register a metric function under ``name``."""

        def decorator(func: Callable[[List[str], List[str]], Dict[str, float]]) -> Callable[[List[str], List[str]], Dict[str, float]]:
            self._metrics[name] = func
            self._descriptions[name] = description
            return func

        return decorator

    def get_metric(self, name: str) -> Callable[[List[str], List[str]], Dict[str, float]] | None:
        """Return the metric function registered under ``name`` if present."""

        return self._metrics.get(name)

    def list_metrics(self) -> Dict[str, str]:
        """Return a mapping of metric names to descriptions."""

        return {name: self._descriptions.get(name, "") for name in self._metrics}


# Global registry and decorator -------------------------------------------------
metric_registry = MetricRegistry()


def register_metric(name: str, description: str = "") -> Callable[[Callable[..., Dict[str, float]]], Callable[..., Dict[str, float]]]:
    """Decorator to register a metric function with :data:`metric_registry`."""

    return metric_registry.register(name, description)


# Metric implementations -------------------------------------------------------


@register_metric("bleu", "SacreBLEU score")
def bleu_metric(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """Compute corpus BLEU using :mod:`evaluate`."""

    metric = evaluate.load("sacrebleu")
    result = metric.compute(predictions=preds, references=[[r] for r in refs])
    return {"bleu": result["score"]}


@register_metric("chrf", "chrF++ score")
def chrf_metric(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """Compute chrF++ using :mod:`evaluate`."""

    metric = evaluate.load("chrf")
    result = metric.compute(predictions=preds, references=refs)
    return {"chrf": result["score"]}


@register_metric("comet", "COMET model evaluation")
def comet_metric(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """Compute COMET score using a pretrained or fine-tuned COMET model."""

    metric = evaluate.load("comet", model_id="Darsala/georgian_comet")
    result = metric.compute(predictions=preds, references=refs, sources=None)
    return {"comet": result["score"]}

