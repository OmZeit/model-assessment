import numpy as np
from typing import List, Dict, Any, Callable

def calculate_ucb(predictions: np.ndarray, uncertainties: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """
    Calculate Upper Confidence Bound (UCB) for active learning.
    Balancing exploitation (high predictions) and exploration (high uncertainty).
    
    predictions: shape (N,)
    uncertainties: shape (N,) - could be variance from ensemble or MC Dropout
    beta: Exploration weight
    """
    return predictions + beta * uncertainties

def dbtl_active_learning_loop(
    candidate_sequences: List[Dict[str, Any]], 
    model_predict_fn: Callable[[List[str]], tuple[np.ndarray, np.ndarray]], 
    batch_size: int = 96,
    beta: float = 2.0
) -> List[Dict[str, Any]]:
    """
    Design-Build-Test-Learn loop selector.
    model_predict_fn should return (mean_predictions, uncertainty_scores)
    """
    sequences = [cand["sequence"] for cand in candidate_sequences]
    means, uncertainties = model_predict_fn(sequences)
    
    ucb_scores = calculate_ucb(means, uncertainties, beta=beta)
    
    # Attach scores
    for i, cand in enumerate(candidate_sequences):
        cand["predicted_yield_mean"] = float(means[i])
        cand["predicted_yield_uncertainty"] = float(uncertainties[i])
        cand["ucb_score"] = float(ucb_scores[i])
        
    return sorted(candidate_sequences, key=lambda x: x["ucb_score"], reverse=True)[:batch_size]
