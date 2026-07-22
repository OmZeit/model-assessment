from typing import List, Dict, Any
from ..ml_core.specialization import leakage_safe_split

def run_homology_benchmark(
    dataset_rows: List[Dict[str, Any]], 
    sequence_col: str, 
    target_col: str, 
    model_train_and_eval_fn: callable,
    homology_threshold: float = 0.70
) -> Dict[str, Any]:
    """
    Validates model generalization by strictly separating train and test sets 
    such that no test sequence shares more than `homology_threshold` k-mer similarity
    with any training sequence.
    """
    splits, diagnostics = leakage_safe_split(
        dataset_rows, 
        sequence_col=sequence_col,
        homology_threshold=homology_threshold,
        val_fraction=0.10,
        test_fraction=0.15
    )
    
    if len(splits["test"]) == 0:
        return {"error": "Dataset is too small or highly homologous to form a valid leakage-safe test split."}
        
    train_data = splits["train"]
    val_data = splits["val"]
    test_data = splits["test"]
    
    metrics = model_train_and_eval_fn(train_data, val_data, test_data, sequence_col, target_col)
    
    return {
        "benchmark_type": "homology_separated_in_silico",
        "homology_threshold": homology_threshold,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "test_metrics": metrics,
        "diagnostics": diagnostics
    }
