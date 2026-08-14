from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

model_path = "models/sbert_model"
output_path = "models/sbert_onnx"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = ORTModelForFeatureExtraction.from_pretrained(
    model_path,
    export=True
)

print("Saving ONNX model...")

model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print("ONNX conversion completed!")
print(f"Saved to: {output_path}")