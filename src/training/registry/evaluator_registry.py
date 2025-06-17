"""
Evaluator Registry

Registry for different evaluation metrics and strategies.
Updated to support COMET metrics that require source sentences.
"""

from typing import Dict, Any, List, Callable, Optional
from .base import BaseRegistry


class EvaluatorRegistry(BaseRegistry):
    """Registry for evaluation metrics and strategies."""

    def validate_component_output(self, output: Any, component_name: str) -> Any:
        """
        Validate that evaluator component returns correct format.

        Args:
            output: Should be a callable that computes metrics
            component_name: Name of the evaluator component

        Returns:
            Validated evaluator function

        Raises:
            ValueError: If output format is incorrect
        """
        if not callable(output):
            raise ValueError(
                f"Evaluator component '{component_name}' must return callable. "
                f"Got: {type(output)}"
            )

        return output

    def create_evaluator(self,
                         evaluator_name: str,
                         config: Dict[str, Any]) -> Callable:
        """
        Create an evaluator using registered components.

        Args:
            evaluator_name: Name of the registered evaluator component
            config: Evaluator configuration parameters

        Returns:
            Configured evaluator function

        Raises:
            ValueError: If evaluator not found or validation fails
        """
        self.validate_component_exists(evaluator_name, "Evaluator")

        evaluator_func = self.get(evaluator_name)
        evaluator = evaluator_func(config)

        return self.validate_component_output(evaluator, evaluator_name)

    def create_combined_evaluator(self,
                                  evaluator_configs: List[Dict[str, Any]]) -> Callable:
        """
        Create a combined evaluator from multiple registered evaluators.

        Args:
            evaluator_configs: List of evaluator configurations, each containing
                              'name' and optional 'config' keys

        Returns:
            Combined evaluator function that computes all metrics

        Example:
            evaluator_configs = [
                {'name': 'sacrebleu', 'config': {}},
                {'name': 'chrf', 'config': {'word_order': 2}},
                {'name': 'georgian_comet', 'config': {'batch_size': 16}}
            ]
        """
        evaluators = []

        for eval_config in evaluator_configs:
            eval_name = eval_config['name']
            eval_params = eval_config.get('config', {})

            evaluator = self.create_evaluator(eval_name, eval_params)
            evaluators.append((eval_name, evaluator))

        def combined_evaluate(predictions: List[str],
                            references: List[str],
                            sources: Optional[List[str]] = None) -> Dict[str, Any]:
            """
            Compute all registered metrics.

            Args:
                predictions: List of predicted translations
                references: List of reference translations
                sources: List of source sentences (optional, required for COMET)

            Returns:
                Dictionary with all computed metrics
            """
            results = {}

            for eval_name, evaluator in evaluators:
                try:
                    # Check if evaluator needs source sentences (like COMET)
                    if 'comet' in eval_name.lower() and sources is not None:
                        # Pass sources to COMET evaluator
                        metric_results = evaluator(predictions, references, sources)
                    else:
                        # Standard evaluators that only need predictions and references
                        metric_results = evaluator(predictions, references)

                    # If evaluator returns dict, merge with prefix
                    if isinstance(metric_results, dict):
                        for key, value in metric_results.items():
                            # Avoid double prefixing for some metrics
                            if key.startswith(eval_name):
                                results[key] = value
                            else:
                                results[f"{eval_name}_{key}"] = value
                    else:
                        # If evaluator returns single value
                        results[eval_name] = metric_results

                except Exception as e:
                    print(f"Warning: Error computing {eval_name}: {e}")
                    results[f"{eval_name}_error"] = str(e)

            return results

        return combined_evaluate


# Global evaluator registry instance
evaluator_registry = EvaluatorRegistry()


def register_evaluator(name: str, description: str = "") -> callable:
    """Convenience function to register an evaluator."""
    return evaluator_registry.register(name, description)