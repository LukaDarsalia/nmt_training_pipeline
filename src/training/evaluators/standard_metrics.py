"""
Standard Evaluation Metrics

Provides standard evaluation metrics for machine translation including
BLEU, CHRF, ROUGE, and METEOR scores.
"""

from typing import Dict, Any, List, Callable
import evaluate
from ..registry.evaluator_registry import register_evaluator


@register_evaluator("bleu", "BLEU score evaluation")
def create_bleu_evaluator(config: Dict[str, Any]) -> Callable:
    """
    Create BLEU score evaluator.

    Args:
        config: Configuration parameters for BLEU

    Returns:
        BLEU evaluation function
    """
    # Load BLEU metric
    bleu_metric = evaluate.load("bleu")

    def evaluate_bleu(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute BLEU score.

        Args:
            predictions: List of predicted translations
            references: List of reference translations

        Returns:
            Dictionary with BLEU score
        """
        # Convert references to list of lists for BLEU
        references_list = [[ref] for ref in references]

        result = bleu_metric.compute(
            predictions=predictions,
            references=references_list
        )

        return {
            "bleu": result["bleu"],
            "bleu_1": result["precisions"][0] if len(result["precisions"]) > 0 else 0.0,
            "bleu_2": result["precisions"][1] if len(result["precisions"]) > 1 else 0.0,
            "bleu_3": result["precisions"][2] if len(result["precisions"]) > 2 else 0.0,
            "bleu_4": result["precisions"][3] if len(result["precisions"]) > 3 else 0.0,
        }

    return evaluate_bleu


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

    def evaluate_sacrebleu(predictions: List[str], references: List[str]) -> Dict[str, float]:
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

        return {"sacrebleu": result["score"]}

    return evaluate_sacrebleu


@register_evaluator("chrf", "CHRF score evaluation")
def create_chrf_evaluator(config: Dict[str, Any]) -> Callable:
    """
    Create CHRF score evaluator.

    Args:
        config: Configuration parameters for CHRF

    Returns:
        CHRF evaluation function
    """
    # Load CHRF metric
    chrf_metric = evaluate.load("chrf")

    # Get configuration parameters
    word_order = config.get("word_order", 0)

    def evaluate_chrf(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute CHRF score.

        Args:
            predictions: List of predicted translations
            references: List of reference translations

        Returns:
            Dictionary with CHRF score
        """
        # Convert references to list of lists for CHRF
        references_list = [[ref] for ref in references]

        result = chrf_metric.compute(
            predictions=predictions,
            references=references_list,
            word_order=word_order
        )

        return {"chrf": result["score"]}

    return evaluate_chrf


@register_evaluator("rouge", "ROUGE score evaluation")
def create_rouge_evaluator(config: Dict[str, Any]) -> Callable:
    """
    Create ROUGE score evaluator.

    Args:
        config: Configuration parameters for ROUGE

    Returns:
        ROUGE evaluation function
    """
    # Load ROUGE metric
    rouge_metric = evaluate.load("rouge")

    def evaluate_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute ROUGE scores.

        Args:
            predictions: List of predicted translations
            references: List of reference translations

        Returns:
            Dictionary with ROUGE scores
        """
        result = rouge_metric.compute(
            predictions=predictions,
            references=references
        )

        return {
            "rouge1": result["rouge1"],
            "rouge2": result["rouge2"],
            "rougeL": result["rougeL"],
            "rougeLsum": result["rougeLsum"]
        }

    return evaluate_rouge


@register_evaluator("meteor", "METEOR score evaluation")
def create_meteor_evaluator(config: Dict[str, Any]) -> Callable:
    """
    Create METEOR score evaluator.

    Args:
        config: Configuration parameters for METEOR

    Returns:
        METEOR evaluation function
    """
    # Load METEOR metric
    meteor_metric = evaluate.load("meteor")

    def evaluate_meteor(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute METEOR score.

        Args:
            predictions: List of predicted translations
            references: List of reference translations

        Returns:
            Dictionary with METEOR score
        """
        result = meteor_metric.compute(
            predictions=predictions,
            references=references
        )

        return {"meteor": result["meteor"]}

    return evaluate_meteor


@register_evaluator("bertscore", "BERTScore evaluation")
def create_bertscore_evaluator(config: Dict[str, Any]) -> Callable:
    """
    Create BERTScore evaluator.

    Args:
        config: Configuration parameters for BERTScore

    Returns:
        BERTScore evaluation function
    """
    # Load BERTScore metric
    bertscore_metric = evaluate.load("bertscore")

    # Get configuration parameters
    model_type = config.get("model_type", "distilbert-base-uncased")
    lang = config.get("lang", "en")

    def evaluate_bertscore(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute BERTScore.

        Args:
            predictions: List of predicted translations
            references: List of reference translations

        Returns:
            Dictionary with BERTScore metrics
        """
        result = bertscore_metric.compute(
            predictions=predictions,
            references=references,
            model_type=model_type,
            lang=lang
        )

        return {
            "bertscore_precision": sum(result["precision"]) / len(result["precision"]),
            "bertscore_recall": sum(result["recall"]) / len(result["recall"]),
            "bertscore_f1": sum(result["f1"]) / len(result["f1"])
        }

    return evaluate_bertscore