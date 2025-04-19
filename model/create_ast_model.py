import torch
from transformers import ASTForAudioClassification


def create_ast_model(num_classes: int, device, weights_path: str | None = None):
    model = ASTForAudioClassification.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    )

    # Replace classifier head with new dense layer
    model.classifier.dense = torch.nn.Linear(
        in_features=768,
        out_features=num_classes,
        bias=True
    )

    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location=device))

    model = model.to(device)
    return model
