import os
import glob
from transformers import LayoutLMv3ForSequenceClassification, LayoutLMv3Processor

HF_REPO_ID = "GouravBanerjee/DocOrNoDoc"

print("🔍 Searching for local model checkpoints...")
checkpoints = glob.glob("./document_classifier_model/checkpoint-*")

if not checkpoints:
    print("❌ No local checkpoints found in ./document_classifier_model/")
    print("Please make sure you have trained your model locally first.")
    exit(1)

# Find the latest checkpoint
best_model_path = max(checkpoints, key=os.path.getctime)
print(f"✅ Found latest checkpoint: {best_model_path}")

print("\n⏳ Loading model and processor into memory...")
# Load the model from the local checkpoint
model = LayoutLMv3ForSequenceClassification.from_pretrained(best_model_path)

# Load the base processor
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

print(f"\n🚀 Pushing model to Hugging Face Hub: {HF_REPO_ID}...")
print("This may take a few minutes depending on your internet connection.")

try:
    # Push model weights
    model.push_to_hub(HF_REPO_ID)
    
    # Push processor (tokenizer and feature extractor)
    processor.push_to_hub(HF_REPO_ID)
    
    print("\n🎉 SUCCESS! Your model has been uploaded to Hugging Face!")
    print(f"You can view it here: https://huggingface.co/{HF_REPO_ID}")

except Exception as e:
    print(f"\n❌ Error pushing to Hugging Face: {e}")
    print("Make sure you have run 'hf auth login' in your terminal and provided a token with WRITE permissions.")
