def compute_monitoring_metrics(system_type, y_true, y_pred):
    """
    Compute the appropriate monitoring metrics for the given system type.
    """
    n = len(y_pred)
    
    if system_type == "regression":
        errors = [abs(p - t) for p, t in zip(y_pred, y_true)]
        mae = sum(errors) / n
        mse = sum([e**2 for e in errors]) / n
        rmse = (mse)**(1/2)
        return [("mae", mae), ("rmse", rmse)]
    elif system_type == "classification":
        tp = sum(1 for p, t in zip(y_pred, y_true) if p == 1 and t == 1)
        tn = sum(1 for p, t in zip(y_pred, y_true) if p == 0 and t == 0)
        fp = sum(1 for p, t in zip(y_pred, y_true) if p == 1 and t == 0)
        fn = sum(1 for p, t in zip(y_pred, y_true) if p == 0 and t == 1)

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / n
        
        return [("accuracy", accuracy), ("f1", f1), ("precision", precision), ("recall", recall)]
    elif system_type == "ranking":
        indices = sorted(range(len(y_pred)), key=lambda i: y_pred[i], reverse=True)
        y_sorted = [y_true[i] for i in indices]
        
        k = 3
        hits = sum(y_sorted[:k])
        
        p_at_3 = hits / k
        
        total_relevant = sum(y_true)
        r_at_3 = (hits / total_relevant) if total_relevant > 0 else 0
        
        return [("precision_at_3", p_at_3), ("recall_at_3", r_at_3)]
    pass