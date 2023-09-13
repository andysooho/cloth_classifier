import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import json
import numpy as np

config_path = "data/config.json"
preprocessor_config_path = "data/preprocessor_config.json"

# Load the data
model_path = "data/pytorch_model.bin"
model = ViTForImageClassification.from_pretrained(model_path, config=config_path)

feature_extractor = ViTImageProcessor.from_pretrained(preprocessor_config_path)

def preprocess_image(image_path: str, feature_extractor) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    pixel_values = feature_extractor(image)["pixel_values"]

    return torch.tensor(np.array(pixel_values))

def classify_image(image_path: str):
    input_image = preprocess_image(image_path, feature_extractor)

    with torch.no_grad():
        output = model(input_image)

    logits = output.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    with open(config_path, 'r') as f:
        config_data = json.load(f)

    id2label = config_data['id2label']
    predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
    predicted_label = id2label[str(predicted_class_idx)]

    return predicted_class_idx, predicted_label
