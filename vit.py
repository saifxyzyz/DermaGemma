from PIL import Image
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
)
import json
import torch

VIT_PATH = "saif0z/vit_skin_classifier" 

print("Loading ViT classifier")
vit_processsor = ViTImageProcessor.from_pretrained(VIT_PATH)
vit_model = ViTForImageClassification.from_pretrained(VIT_PATH).to("cpu")
vit_model.eval()

id2label = vit_model.config.id2label

print(f"ViT loaded - knows {len(id2label)} conditions")
formatted_labels = "\n".join(id2label.values())
formatted_labels = "\n".join(id2label.values())
# print(formatted_labels)
def classify_image(image_path, top_k=3):
    if isinstance(image_path, str):
        image = Image.open(image_path).convert("RGB")
    else:
        image = image_path.convert("RGB")
    inputs = vit_processsor(images=image, return_tensors="pt").to("cpu")
    
    with torch.no_grad():
        outputs = vit_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    top_probs, top_indices = probs.topk(top_k)

    return [
        {
            "condition": id2label[idx.item()].replace("_", " "),
            "confidence": prob.item()
        }
        for prob, idx in zip(top_probs, top_indices)
    ]


print(classify_image("test_images/acne_vulgaris_black.jpg", 3))
