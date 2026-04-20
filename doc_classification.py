# -*- coding: utf-8 -*-
"""
LayoutLMv3 Master Script: 3-Class Classification
Goal: Differentiate between Other, Sales Invoices, and Sales Returns
Techniques: EasyOCR, Keyword Masking (Blindfold), Multimodal Deep Learning
"""

# ==========================================
# 1. INSTALLATIONS & IMPORTS
# ==========================================
print("⏳ Setting up environment...")

import os, glob, torch, easyocr, evaluate, shutil, numpy as np, pandas as pd
from PIL import Image
from pdf2image import convert_from_path
from datasets import Dataset, Features, Sequence, Value, ClassLabel
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForSequenceClassification,
    TrainingArguments,
    Trainer
)

# ==========================================
# 2. CONFIGURATION & MASKING LOGIC
# ==========================================
# Labels
ID2LABEL = {0: "NON_ACCOUNTING", 1: "SALES_INVOICE", 2: "SALES_RETURN"}
LABEL2ID = {"NON_ACCOUNTING": 0, "SALES_INVOICE": 1, "SALES_RETURN": 2}

# The "Blindfold" List: Force model to look at layout, not these words
IGNORE_LIST = ["BUSY", "TALLY", "VYAPAR", "MARG", "ZOHO", "SOFTWARE", "INVENTORY"]

processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
reader = easyocr.Reader(['en'])


def mask_text(text_list):
    """Replaces brand names with 'DOCUMENT' to prevent brand overfitting"""
    return ["DOCUMENT" if any(b in w.upper() for b in IGNORE_LIST) else w for w in text_list]


# ==========================================
# 3. DATA LOADING (SCAN FOLDERS)
# ==========================================
def load_data_from_folders():
    print("📂 Scanning folders for training data...")
    data_list = []
    # Define folder structure
    dirs = {
        1: "Dataset/Sales Invoice",
        2: "Dataset/Sales Return",
        0: "Dataset/Non Accounting Docs"
    }

    for label_id, folder_path in dirs.items():
        if not os.path.exists(folder_path):
            print(f"⚠️ Warning: Folder {folder_path} not found. Skipping.")
            continue

        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg'))]
        print(f"  - Found {len(files)} files in {ID2LABEL[label_id]}")

        for f in files:
            data_list.append({"file_name": f, "label": label_id, "path": os.path.join(folder_path, f)})

    return pd.DataFrame(data_list)


# ==========================================
# 4. PREPROCESSING (PDF -> OCR -> MASKING)
# ==========================================
def preprocess_data(examples):
    images, words_list, boxes_list, labels = [], [], [], []

    for i, path in enumerate(examples['path']):
        try:
            # 1. Convert to Image
            if path.lower().endswith(".pdf"):
                img = convert_from_path(path, first_page=1, last_page=1, dpi=200)[0].convert("RGB")
            else:
                img = Image.open(path).convert("RGB")

            # 2. Get OCR with EasyOCR
            res = reader.readtext(np.array(img))
            if not res: continue

            # 3. Extract and Mask Text
            raw_words = [r[1] for r in res]
            masked_words = mask_text(raw_words)

            # 4. Normalize Boxes (0-1000)
            w, h = img.size
            boxes = [[int(1000 * (r[0][0][0] / w)), int(1000 * (r[0][0][1] / h)),
                      int(1000 * (r[0][2][0] / w)), int(1000 * (r[0][2][1] / h))] for r in res]

            images.append(img)
            words_list.append(masked_words)
            boxes_list.append(boxes)
            labels.append(examples['label'][i])
        except Exception as e:
            print(f"Skipping {path}: {e}")

    if not images: return {"input_ids": [], "bbox": [], "pixel_values": [], "labels": []}

    encoding = processor(
        images, words_list, boxes=boxes_list,
        max_length=512, truncation=True, padding="max_length"
    )
    encoding["labels"] = labels
    return encoding


# ==========================================
# 5. TRAINING PHASE
# ==========================================
def train_model():
    df = load_data_from_folders()
    if df.empty: return print("❌ No data found to train on!")

    # Convert to HuggingFace Dataset
    raw_ds = Dataset.from_pandas(df)
    ds_split = raw_ds.train_test_split(test_size=0.15, seed=42)

    print("🎨 Transforming dataset (this takes time)...")
    processed_ds = ds_split.map(preprocess_data, batched=True, batch_size=2, remove_columns=raw_ds.column_names)

    print("🏋️ Training the 3-Class Brain...")
    model = LayoutLMv3ForSequenceClassification.from_pretrained(
        "microsoft/layoutlmv3-base", num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    )

    training_args = TrainingArguments(
        output_dir="document_classifier_v3",
        num_train_epochs=10,  # More epochs for 3 classes
        per_device_train_batch_size=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none"
    )

    metric = evaluate.load("accuracy")

    def compute_metrics(p):
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=processed_ds["train"], eval_dataset=processed_ds["test"],
        compute_metrics=compute_metrics
    )

    trainer.train()
    print("✅ Model Training Complete!")


# ==========================================
# 6. INFERENCE (TESTING) PHASE
# ==========================================
def run_batch_test():
    checkpoints = glob.glob("./document_classifier_v3/checkpoint-*")
    if not checkpoints: return print("❌ No model found! Train first.")

    best_model_path = max(checkpoints, key=os.path.getctime)
    print(f"🧠 Loading Brain: {best_model_path}")
    model = LayoutLMv3ForSequenceClassification.from_pretrained(best_model_path)

    test_folder = "Dataset/Test"
    print(f"📂 Testing documents in {test_folder}...\n")

    for filename in sorted(os.listdir(test_folder)):
        path = os.path.join(test_folder, filename)
        if filename.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
            try:
                img = convert_from_path(path, first_page=1, last_page=1, dpi=200)[0].convert(
                    "RGB") if filename.endswith(".pdf") else Image.open(path).convert("RGB")

                # Live OCR + Masking
                res = reader.readtext(np.array(img))
                words = mask_text([r[1] for r in res])
                w, h = img.size
                boxes = [[int(1000 * (r[0][0][0] / w)), int(1000 * (r[0][0][1] / h)),
                          int(1000 * (r[0][2][0] / w)), int(1000 * (r[0][2][1] / h))] for r in res]

                inputs = processor(img, text=words, boxes=boxes, return_tensors="pt", truncation=True, max_length=512,
                                   padding="max_length")

                with torch.no_grad():
                    outputs = model(**inputs)

                pred = torch.argmax(outputs.logits, dim=-1).item()
                conf = torch.softmax(outputs.logits, dim=-1)[0][pred].item()

                # Label mapping for output
                label_name = ID2LABEL[pred]
                icon = "📄" if pred == 0 else ("✅" if pred == 1 else "🔄")
                print(f"{icon} {filename:35} -> {label_name} ({conf * 100:.1f}%)")
            except Exception as e:
                print(f"❌ Error {filename}: {e}")


# ==========================================
# 7. EXECUTION (The "Run" Switch)
# ==========================================
if __name__ == "__main__":
    # STEP 1: Train the new model
    train_model()

    # STEP 2: Run the test
    run_batch_test()
