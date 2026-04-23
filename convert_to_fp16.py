import os
import glob
import torch
from transformers import LayoutLMv3ForSequenceClassification, LayoutLMv3Processor

# --- CONFIGURATION ---
# The Hugging Face Hub repository ID where the model will be pushed.
HF_REPO_ID = "GouravBanerjee/DocOrNoDoc"
# The local folder where your last successful training run was saved.
MODEL_SOURCE_DIR = "./document_classifier_model_fp16"


def convert_and_push_fp16():
    """
    Finds the latest full-precision model, converts it to FP16,
    and pushes the smaller version to a dedicated 'fp16' branch on the Hub.
    """
    print("--- 🚀 Starting FP16 Conversion and Upload ---")

    # 1. Find the latest local checkpoint
    print(f"🔍 Searching for checkpoints in '{MODEL_SOURCE_DIR}'...")
    checkpoints = glob.glob(f"{MODEL_SOURCE_DIR}/checkpoint-*")
    if not checkpoints:
        print(f"❌ No checkpoints found in {MODEL_SOURCE_DIR}.")
        print("Please ensure you have run a successful training session first.")
        return

    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    print(f"✅ Found latest checkpoint: {latest_checkpoint}")

    # 2. Load the full-precision model (approx. 500MB)
    print("\n⏳ Loading full-precision model from disk...")
    try:
        model = LayoutLMv3ForSequenceClassification.from_pretrained(latest_checkpoint)
        processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    except Exception as e:
        print(f"❌ Error loading the model: {e}")
        return

    # 3. Convert the model to FP16 (half-precision)
    print("⚡ Converting model to FP16 in memory...")
    model.half()
    print("✅ Model converted successfully. Size is now ~250MB.")

    # 4. Push the smaller FP16 model and processor to the Hub
    print(f"\n🚀 Pushing FP16 model to '{HF_REPO_ID}' on branch 'fp16'...")
    print("This may take a minute depending on your internet connection.")
    try:
        # Push the model to the 'fp16' branch (revision)
        model.push_to_hub(HF_REPO_ID, revision="fp16", commit_message="Add fp16 quantized model")
        
        # Push the processor to the same branch for consistency
        processor.push_to_hub(HF_REPO_ID, revision="fp16", commit_message="Add processor for fp16 model")
        
        print("\n🎉 SUCCESS! Your half-sized FP16 model is now on the Hub.")
        print("You can load it from anywhere using:")
        print(f"  model = LayoutLMv3ForSequenceClassification.from_pretrained('{HF_REPO_ID}', revision='fp16')")
        print(f"\nView it online: https://huggingface.co/{HF_REPO_ID}/tree/fp16")

    except Exception as e:
        print(f"\n❌ An error occurred during the upload: {e}")
        print("Please ensure you are logged in via 'hf auth login' and have write permissions.")


if __name__ == "__main__":
    convert_and_push_fp16()
