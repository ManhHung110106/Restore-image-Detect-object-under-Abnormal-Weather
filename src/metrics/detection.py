# Detection metrics are typically extracted from ultralytics output dictionaries directly.
# This module provides a helper to standardize the result formatting.

def extract_detection_metrics(val_results) -> dict:
    """
    Extract relevant metrics from ultralytics validation results object.
    """
    # Note: val_results is typically an ultralytics.engine.results.Results object from model.val()
    metrics = val_results.results_dict
    
    # We want: mAP50, mAP50-95, precision, recall, F1, per-class AP
    # ultralytics dict keys typically are:
    # 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)'
    
    # Try different key formats just in case
    def get_key(k_list):
        for k in k_list:
            if k in metrics:
                return metrics[k]
        return 0.0

    p = get_key(['metrics/precision(B)', 'precision'])
    r = get_key(['metrics/recall(B)', 'recall'])
    map50 = get_key(['metrics/mAP50(B)', 'mAP50'])
    map50_95 = get_key(['metrics/mAP50-95(B)', 'mAP50-95'])
    
    # F1-score
    eps = 1e-7
    f1 = 2 * p * r / (p + r + eps)
    
    res = {
        "map50": map50,
        "map50_95": map50_95,
        "precision": p,
        "recall": r,
        "f1": f1
    }
    
    # Per-class AP (if available from val_results.box)
    try:
        if hasattr(val_results, 'box') and val_results.box is not None:
            # val_results.box.ap50 is an array of AP50 for each class
            # val_results.names is the dict of class names
            ap50_per_class = val_results.box.ap50
            class_ids = val_results.box.ap_class_index
            names = val_results.names
            
            for i, c_id in enumerate(class_ids):
                name = names.get(c_id, f"class_{c_id}")
                res[f"ap_{name}"] = ap50_per_class[i]
    except Exception as e:
        print(f"Warning: Could not extract per-class AP: {e}")
        
    return res
