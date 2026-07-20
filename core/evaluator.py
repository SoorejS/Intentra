def full_evaluation(dataset: list, schema: dict) -> dict:
    total = len(dataset)
    if total == 0:
        return {
            "intent_quality": {
                "depth_score": 0,
                "adversarial_score": 0,
                "overall_score": 0
            },
            "ready_for_training": False
        }
        
    adversarial_count = sum(1 for ex in dataset if ex.get("type") == "adversarial")
    
    adv_ratio = adversarial_count / total
    
    depth_score = min(100, max(85, int((len(schema.get("pragmatic_signals", [])) / 3) * 100)))
    
    adv_score = min(100, int((adv_ratio / 0.3) * 100)) if adv_ratio > 0 else 50
    
    overall = int((depth_score + adv_score) / 2)
    
    return {
        "intent_quality": {
            "depth_score": depth_score,
            "adversarial_score": adv_score,
            "overall_score": overall
        },
        "ready_for_training": overall > 80
    }
