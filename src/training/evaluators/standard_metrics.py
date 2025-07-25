"""
Standard Evaluation Metrics

Provides essential evaluation metrics for machine translation:
SacreBLEU and chrF++ only.
"""

from typing import Dict, Any, List, Callable, Optional
import evaluate
from ..registry.evaluator_registry import register_evaluator


@register_evaluator("sacrebleu", "SacreBLEU score evaluation")
def create_sacrebleu_evaluator(config: Dict[str, Any]) -> Callable:
    """
    Create SacreBLEU score evaluator.

    Args:
        config: Configuration parameters for SacreBLEU

    Returns:
        SacreBLEU evaluation function
    """
    # Load SacreBLEU metric
    sacrebleu_metric = evaluate.load("sacrebleu")

    def evaluate_sacrebleu(predictions: List[str], references: List[str], sources: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Compute SacreBLEU score.

        Args:
            predictions: List of predicted translations
            references: List of reference translations

        Returns:
            Dictionary with SacreBLEU score
        """
        # Convert references to list of lists for SacreBLEU
        references_list = [[ref] for ref in references]

        result = sacrebleu_metric.compute(
            predictions=predictions,
            references=references_list
        )

        if result is None:
            raise ValueError("SacreBLEU result is None")

        return {"sacrebleu": result["score"]}

    return evaluate_sacrebleu


@register_evaluator("chrf", "chrF++ score evaluation")
def create_chrf_evaluator(config: Dict[str, Any]) -> Callable:
    """
    Create chrF++ score evaluator.

    Args:
        config: Configuration parameters for chrF++
            - word_order: int, default 2 (for chrF++)

    Returns:
        chrF++ evaluation function
    """
    # Load CHRF metric
    chrf_metric = evaluate.load("chrf")

    # Get configuration parameters - default to chrF++ (word_order=2)
    word_order = config.get("word_order", 2)

    def evaluate_chrf(predictions: List[str], references: List[str], sources: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Compute chrF++ score.

        Args:
            predictions: List of predicted translations
            references: List of reference translations

        Returns:
            Dictionary with chrF++ score
        """
        # Convert references to list of lists for CHRF
        references_list = [[ref] for ref in references]

        result = chrf_metric.compute(
            predictions=predictions,
            references=references_list,
            word_order=word_order
        )

        if result is None:
            raise ValueError("CHRF result is None")

        return {"chrf++": result["score"]}

    return evaluate_chrf
