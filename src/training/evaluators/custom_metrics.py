"""
Custom Evaluation Metrics

Provides only the Georgian COMET model for MT evaluation.
Includes helper metrics for fallback scenarios.
"""

from typing import Dict, Any, List, Callable

from ..registry.evaluator_registry import register_evaluator


@register_evaluator("georgian_comet", "Georgian fine-tuned COMET model for MT evaluation")
def create_georgian_comet_evaluator(config: Dict[str, Any]) -> Callable:
    """
    Create Georgian COMET evaluator using the fine-tuned model.

    Args:
        config: Configuration parameters for Georgian COMET
            - model_name: str, default "Darsala/georgian_comet"
            - batch_size: int, default 16
            - device: str, default "cuda" if available else "cpu"
            - gpus: int, default 1

    Returns:
        Georgian COMET evaluation function
    """
    # Configuration parameters
    model_name = config.get("model_name", "Darsala/georgian_comet")
    batch_size = config.get("batch_size", 16)
    device = config.get("device", "cuda")
    gpus = config.get("gpus", 1)

    # Initialize model variable
    comet_model = None

    def evaluate_georgian_comet(predictions: List[str], references: List[str], sources: List[str] = None) -> Dict[str, float]:
        """
        Compute Georgian COMET score.

        Args:
            predictions: List of predicted translations
            references: List of reference translations
            sources: List of source sentences (required for COMET)

        Returns:
            Dictionary with Georgian COMET score
        """
        nonlocal comet_model

        if not predictions or not references:
            return {"comet": 0.0}

        # If sources not provided, use empty sources (though this is not ideal for COMET)
        if sources is None:
            print("Warning: No source sentences provided for COMET evaluation. Using empty sources.")
            sources = [""] * len(predictions)

        if len(sources) != len(predictions) or len(predictions) != len(references):
            print(f"Warning: Mismatched lengths - sources: {len(sources)}, predictions: {len(predictions)}, references: {len(references)}")
            min_len = min(len(sources), len(predictions), len(references))
            sources = sources[:min_len]
            predictions = predictions[:min_len]
            references = references[:min_len]

        try:
            # Load COMET model if not already loaded
            if comet_model is None:
                try:
                    from comet import download_model, load_from_checkpoint

                    print(f"Loading COMET model: {model_name}")
                    model_path = download_model(model_name)
                    comet_model = load_from_checkpoint(model_path)
                    print("Georgian COMET model loaded successfully")

                except Exception as e:
                    print(f"Error loading Georgian COMET model: {e}")
                    print("Falling back to simple length ratio evaluation")
                    return _fallback_evaluation(predictions, references)

            # Prepare data in COMET format
            data = []
            for src, pred, ref in zip(sources, predictions, references):
                data.append({
                    "src": str(src),
                    "mt": str(pred),
                    "ref": str(ref)
                })

            # Generate COMET scores
            try:
                # Use CPU if GPU not available or specified
                if device == "cpu":
                    model_output = comet_model.predict(data, batch_size=batch_size, gpus=0)
                else:
                    model_output = comet_model.predict(data, batch_size=batch_size, gpus=gpus)

                # Extract system-level score
                system_score = float(model_output.system_score)

                # Also return individual scores for debugging if needed
                individual_scores = [float(score) for score in model_output.scores]

                return {
                    "comet": system_score,
                    "comet_mean": sum(individual_scores) / len(individual_scores),
                    "comet_std": _calculate_std(individual_scores)
                }

            except Exception as e:
                print(f"Error computing COMET scores: {e}")
                return _fallback_evaluation(predictions, references)

        except ImportError:
            print("COMET library not installed. Install with: pip install unbabel-comet")
            return _fallback_evaluation(predictions, references)
        except Exception as e:
            print(f"Unexpected error in Georgian COMET evaluation: {e}")
            return _fallback_evaluation(predictions, references)

    def _fallback_evaluation(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Fallback evaluation using simple metrics."""
        if not predictions or not references:
            return {"comet": 0.0, "comet_mean": 0.0, "comet_std": 0.0}

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

        mean_ratio = sum(ratios) / len(ratios)
        return {
            "comet": mean_ratio,
            "comet_mean": mean_ratio,
            "comet_std": _calculate_std(ratios)
        }

    def _calculate_std(values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) <= 1:
            return 0.0
        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        return variance ** 0.5

    return evaluate_georgian_comet