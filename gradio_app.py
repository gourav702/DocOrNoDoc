import os
import glob
import numpy as np
import torch
import easyocr
from PIL import Image
from pdf2image import convert_from_path
import gradio as gr
from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification

# --- SETUP & MODEL LOADING ---
print("⏳ Loading Models... This may take a minute.")
reader = easyocr.Reader(['en'])

# Automatically find the latest checkpoint in the local directory (from main.py)
checkpoints = glob.glob("./document_classifier_model/checkpoint-*")
if not checkpoints:
    print("❌ No trained model found! Please run the training in main.py first.")
    exit(1)

MODEL_PATH = max(checkpoints, key=os.path.getctime)
print(f"🧠 Loading Local Brain from: {MODEL_PATH}")

processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = LayoutLMv3ForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval() # Set to evaluation mode

def get_ocr_data(img):
    """Your original OCR logic"""
    results = reader.readtext(np.array(img))
    width, height = img.size
    words, boxes = [], []
    for (bbox, text, prob) in results:
        words.append(text)
        x0, y0 = bbox[0]
        x1, y1 = bbox[2]
        boxes.append([
            max(0, min(1000, int(1000 * (x0 / width)))),
            max(0, min(1000, int(1000 * (y0 / height)))),
            max(0, min(1000, int(1000 * (x1 / width)))),
            max(0, min(1000, int(1000 * (y1 / height))))
        ])
    return words, boxes

def predict_document(file_obj):
    if file_obj is None:
        return {"Error": "No file uploaded."}

    try:
        # Extract the file path depending on how Gradio passes the file
        file_path = file_obj.name if hasattr(file_obj, 'name') else file_obj
        
        # 1. Handle PDF vs Image
        if str(file_path).lower().endswith(".pdf"):
            # Convert first page of PDF to Image
            images = convert_from_path(file_path, first_page=1, last_page=1, dpi=200)
            img = images[0].convert("RGB")
        else:
            img = Image.open(file_path).convert("RGB")

        # 2. Get OCR
        words, boxes = get_ocr_data(img)
        if not words:
            return {"Error": "OCR failed: No text detected in document."}

        # 3. Prepare for LayoutLMv3
        inputs = processor(
            img,
            text=words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        )

        # 4. Inference
        with torch.no_grad():
            outputs = model(**inputs)

        prediction = torch.argmax(outputs.logits, dim=-1).item()
        probs = torch.softmax(outputs.logits, dim=-1)
        confidence = probs[0][prediction].item()

        # 5. Format Results
        label = "ACCOUNTING (BUSY)" if prediction == 1 else "OTHER / NON-ACCOUNTING"
        
        return {
            "Classification": label,
            "Confidence": f"{confidence*100:.2f}%",
            "Details": f"Detected {len(words)} words on the first page."
        }

    except Exception as e:
        return {"Error": f"Error processing file: {str(e)}"}

# --- GRADIO INTERFACE ---
interface = gr.Interface(
    fn=predict_document,
    inputs=gr.File(label="Upload Document (PDF or Image)", file_types=[".pdf", ".png", ".jpg", ".jpeg"]),
    outputs=gr.JSON(label="Analysis Result"),
    title="AI Document Classifier",
    description="Upload an accounting document (invoice, receipt, bank statement) or a general document to classify it.",
    theme="soft"
)

if __name__ == "__main__":
    # Launching the interface locally
    interface.launch(share=False)
