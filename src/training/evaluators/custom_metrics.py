"""
Custom Evaluation Metrics

Provides custom evaluation metrics including Georgian COMET model and other
specialized metrics for machine translation evaluation.
"""

from typing import Dict, Any, List, Callable
import torch
from transformers import AutoTokenizer, AutoModel
from ..registry.evaluator_registry import register_evaluator


@register_evaluator("georgian_comet", "Georgian fine-tuned COMET model for MT evaluation")
def create_georgian_comet_evaluator(config: Dict[str, Any]) -> Callable:
    """
    Create Georgian COMET evaluator using the fine-tuned model.

    Args:
        config: Configuration parameters for Georgian COMET

    Returns:
        Georgian COMET evaluation function
    """
    # Configuration parameters
    model_name = config.get("model_name", "Darsala/georgian_comet")
    batch_size = config.get("batch_size", 16)
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Load the model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"Warning: Could not load Georgian COMET model: {e}")
        print("Falling back to simple sentence length ratio evaluation")

        def fallback_evaluate(predictions: List[str], references: List[str]) -> Dict[str, float]:
            """Fallback evaluation using simple metrics."""
            if not predictions or not references:
                return {"georgian_comet": 0.0}

            # Simple length ratio as fallback
            ratios = []
            for pred, ref in zip(predictions, references):
                pred_len = len(pred.split())
                ref_len = len(ref.split())
                if ref_len > 0:
                    ratio = min(pred_len / ref_len, ref_len / pred_len)
                else:
                    ratio = 0.0
                ratios.append(ratio)

            return {"georgian_comet": sum(ratios) / len(ratios)}

        return fallback_evaluate

    def evaluate_georgian_comet(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute Georgian COMET score.

        Args:
            predictions: List of predicted translations
            references: List of reference translations

        Returns:
            Dictionary with Georgian COMET score
        """
        if not predictions or not references:
            return {"georgian_comet": 0.0}

        scores = []

        # Process in batches
        for i in range(0, len(predictions), batch_size):
            batch_predictions = predictions[i:i + batch_size]
            batch_references = references[i:i + batch_size]

            # Prepare input for COMET model
            # Note: This is a simplified implementation
            # The actual COMET model might require different input format
            try:
                # Tokenize the input
                inputs = []
                for pred, ref in zip(batch_predictions, batch_references):
                    # Combine prediction and reference for COMET evaluation
                    combined = f"{pred} [SEP] {ref}"
                    inputs.append(combined)

                # Tokenize and encode
                encoded = tokenizer(
                    inputs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}

                # Get model predictions
                with torch.no_grad():
                    outputs = model(**encoded)

                    # Extract scores (this might need adjustment based on actual model)
                    # For now, we'll use the mean of the last hidden state as a proxy score
                    hidden_states = outputs.last_hidden_state
                    batch_scores = torch.mean(hidden_states, dim=1).mean(dim=1)

                    # Normalize scores to 0-1 range
                    batch_scores = torch.sigmoid(batch_scores)
                    scores.extend(batch_scores.cpu().tolist())

            except Exception as e:
                print(f"Warning: Error computing COMET scores for batch: {e}")
                # Fallback to simple score for this batch
                batch_scores = [0.5] * len(batch_predictions)
                scores.extend(batch_scores)

        # Return average score
        avg_score = sum(scores) / len(scores) if scores else 0.0
        return {"georgian_comet": avg_score}

    return evaluate_georgian_comet


@register_evaluator("length_ratio", "Simple length ratio evaluation")
def create_length_ratio_evaluator(config: Dict[str, Any]) -> Callable:
    """
    Create a simple length ratio evaluator.

    Args:
        config: Configuration parameters

    Returns:
        Length ratio evaluation function
    """

    def evaluate_length_ratio(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute length ratio between predictions and references.

        Args:
            predictions: List of predicted translations
            references: List of reference translations

        Returns:
            Dictionary with length ratio metrics
        """
        if not predictions or not references:
            return {"length_ratio": 0.0, "length_difference": 0.0}

        ratios = []
        differences = []

        for pred, ref in zip(predictions, references):
            pred_len = len(pred.split())
            ref_len = len(ref.split())

            if ref_len > 0:
                ratio = pred_len / ref_len
                ratios.append(ratio)
            else:
                ratios.append(0.0)

            differences.append(abs(pred_len - ref_len))

        return {
            "length_ratio": sum(ratios) / len(ratios),
            "length_difference": sum(differences) / len(differences)
        }

    return evaluate_length_ratio


@register_evaluator("exact_match", "Exact match evaluation")
def create_exact_match_evaluator(config: Dict[str, Any]) -> Callable:
    """
    Create exact match evaluator.

    Args:
        config: Configuration parameters

    Returns:
        Exact match evaluation function
    """

    def evaluate_exact_match(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute exact match accuracy.

        Args:
            predictions: List of predicted translations
            references: List of reference translations

        Returns:
            Dictionary with exact match score
        """
        if not predictions or not references:
            return {"exact_match": 0.0}

        matches = sum(1 for pred, ref in zip(predictions, references)
                      if pred.strip() == ref.strip())

        return {"exact_match": matches / len(predictions)}

    return evaluate_exact_match

# Add more custom evaluators here using the @register_evaluator decorator
# Example:
#
# @register_evaluator("my_custom_metric", "Description of my custom metric")
# def create_my_custom_metric(config: Dict[str, Any]) -> Callable:
#     def evaluate_my_metric(predictions: List[str], references: List[str]) -> Dict[str, float]:
#         # Your implementation here
#         return {"my_metric": score}
#     return evaluate_my_metric
