"""Validation-only checkpoint selection with explicit tie breakers."""


def selection_key(metrics, criterion):
    classification = metrics["gold_aspect_sentiment"]
    accuracy, macro_f1 = classification["accuracy"], classification["macro_f1"]
    if criterion == "gold_accuracy":
        return accuracy, macro_f1
    if criterion == "gold_macro_f1":
        return macro_f1, accuracy
    if criterion == "end_to_end_f1":
        return metrics["end_to_end"]["f1"], macro_f1, accuracy
    raise ValueError(f"Unknown selection criterion: {criterion}")


def checkpoint_directory(criterion, primary):
    if criterion == primary:
        return "best"
    return {"gold_accuracy": "best_accuracy", "gold_macro_f1": "best_macro_f1"}[criterion]
