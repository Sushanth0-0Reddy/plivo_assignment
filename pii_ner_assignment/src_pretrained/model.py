from transformers import AutoModelForTokenClassification


def create_model(model_name: str):
    """Load the pretrained token classification model without changing its head."""
    return AutoModelForTokenClassification.from_pretrained(model_name)
