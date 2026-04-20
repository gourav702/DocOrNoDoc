# -*- coding: utf-8 -*-
"""
This script is adapted from a Google Colab notebook.
Original file is located at
    https://colab.research.google.com/drive/176EFt7781CdqQsW1mvX-70QyBQOcGsJ1
"""

# Consolidated Imports
import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import easyocr
import evaluate
import ssl
from datasets import load_from_disk

from PIL import Image
from pdf2image import convert_from_path
from datasets import Dataset, Features, Sequence, Value, ClassLabel
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForSequenceClassification,
    TrainingArguments,
    Trainer
)

# Mac SSL Fix for easyocr downloading models
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# --- Section 1: Initial Data Extraction (Commented Out) ---
# This section was likely used to generate the 'master_dataset.json' file.
# Since you already have this file, the code is commented out to prevent accidental execution.

"""EDA AND PRE PROCESSING DATA"""
# print("--- 🚀 BUSINESS DOCUMENT HEALTH REPORT ---")
# 
# # 1. Load the actual JSON data
# with open('Dataset/master_dataset.json', 'r') as f:
#     data = json.load(f)
# 
# df = pd.DataFrame(data)
# 
# # Ensure 'text' column handles potential missing or non-list values gracefully
# df['text'] = df['text'].apply(lambda x: x if isinstance(x, list) else [])
# 
# # 2. Advanced Health Checks
# df['word_count'] = df['text'].apply(len)
# failed_ocr = df[df['word_count'] < 5]
# long_docs = df[df['word_count'] > 450]
# 
# print(f"Total Pages Processed: {len(df)}")
# print(f"OCR Failures (Empty):  {len(failed_ocr)}")
# print(f"Long Docs (>450 words): {len(long_docs)} (Risk of truncation)")
# 
# # 3. Class Balance Check
# print("\nRows per Category (1=Accounting, 0=Other):")
# print(df['label'].value_counts())
# 
# # 4. Visualizing Sequence Lengths
# plt.figure(figsize=(10, 5))
# plt.hist(df[df['label'] == 1]['word_count'], bins=20, alpha=0.5, label='Accounting', color='green')
# plt.hist(df[df['label'] == 0]['word_count'], bins=20, alpha=0.5, label='Other', color='red')
# plt.axvline(x=512, color='blue', linestyle='--', label='Model Limit (512)')
# plt.title("Distribution of Word Counts per Document")
# plt.xlabel("Number of Words")
# plt.ylabel("Number of Documents")
# plt.legend()
# plt.savefig('word_count_distribution.png') # Saved to file instead of blocking the script!
# plt.close()
# print("📊 Graph saved as 'word_count_distribution.png'")
# 
# """remove duplicates"""
# print("\n--- 👯 Duplicate Removal ---")
# print("Converting lists to strings for comparison (This may take a moment on large datasets)...")
# # Convert the list of words into a single string to compare them
# df['temp_text_string'] = df['text'].apply(lambda x: " ".join(x))
# count_before = len(df)
# print("Dropping duplicates...")
# df = df.drop_duplicates(subset=['temp_text_string'], keep='first')
# df = df.drop(columns=['temp_text_string'])
# count_after = len(df)
# print(f"Initial rows: {count_before}")
# print(f"Duplicates removed: {count_before - count_after}")
# print(f"Final unique rows: {count_after}")


"""Multimodal Tokenization"""
print("\n--- 🤖 Multimodal Tokenization ---")
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

# """Test Train Split"""
# print("\n--- 🔪 Test Train Split ---")
# df_for_dataset = df[['text', 'boxes', 'label', 'file_name']].copy()
# 
# features = Features({
#     'text': Sequence(Value('string')),
#     'boxes': Sequence(Sequence(Value('int64'))),
#     'label': ClassLabel(names=['other', 'accounting']),
#     'file_name': Value('string')
# })
# 
# raw_dataset = Dataset.from_pandas(df_for_dataset, features=features, preserve_index=False)
# dataset_split = raw_dataset.train_test_split(test_size=0.2, seed=42)
# 
# print(f"✅ Split Complete!")
# print(f"📖 Study Set (Training): {len(dataset_split['train'])} documents")
# print(f"📝 Exam Set (Testing):   {len(dataset_split['test'])} documents")


"""Transformation"""
# print("\n--- 🎨 Transformation ---")
# print("NOTE: This step opens and processes every document image. It might take several minutes!")
# 
# def preprocess_data(examples):
#     images = []
#     valid_indices = []
# 
#     for i, filename in enumerate(examples['file_name']):
#         print(f"  [Processing file] {filename}")
#         
#         path_busy = f"Dataset/Accounting Docs/{filename}"
#         path_non = f"Dataset/Non Accounting Docs/{filename}"
#         target_path = path_busy if os.path.exists(path_busy) else path_non
# 
#         if os.path.exists(target_path):
#             try:
#                 if target_path.lower().endswith(".pdf"):
#                     pages = convert_from_path(target_path, first_page=1, last_page=1)
#                     img = pages[0].convert("RGB")
#                 else:
#                     img = Image.open(target_path).convert("RGB")
# 
#                 images.append(img)
#                 valid_indices.append(i)
#             except Exception as e:
#                 print(f"  ⚠️ Skipping {filename} due to error: {e}")
#         else:
#             print(f"  ⚠️ Could not find file on disk: {filename}")
# 
#     if not images:
#         return {"input_ids": [], "bbox": [], "pixel_values": [], "labels": []}
# 
#     encoding = processor(
#         images,
#         [examples['text'][i] for i in valid_indices],
#         boxes=[examples['boxes'][i] for i in valid_indices],
#         max_length=512,
#         truncation=True,
#         padding="max_length"
#     )
# 
#     encoding["labels"] = [examples['label'][i] for i in valid_indices]
#     return encoding
# 
# processed_dataset = dataset_split.map(
#     preprocess_data,
#     batched=True,
#     batch_size=2,
#     remove_columns=dataset_split["train"].column_names
# )

"""Save and Load Transformed Data to skip the mapping phase later (optional)"""
# You can uncomment this to save/load from disk, avoiding reprocessing.
# processed_dataset.save_to_disk("Dataset/processed_dataset")
print("\n--- 📂 Loading Processed Dataset ---")
try:
    processed_dataset = load_from_disk("Dataset/processed_dataset")
    print("✅ Loaded processed dataset successfully.")
except Exception as e:
    print(f"❌ Failed to load processed dataset. You must run the Transformation step at least once. Error: {e}")
    exit(1)


"""Training"""
# I am keeping this commented out for now since the training already finished successfully
# and saved the model! Let's just run the Testing phase to see the results.
# print("\n--- 🏋️ Training ---")
# model = LayoutLMv3ForSequenceClassification.from_pretrained(
#     "microsoft/layoutlmv3-base",
#     num_labels=2
# )
# 
# metric = evaluate.load("accuracy")
# 
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)
# 
# training_args = TrainingArguments(
#     output_dir="document_classifier_model",
#     learning_rate=2e-5,
#     per_device_train_batch_size=2,
#     num_train_epochs=1, 
#     weight_decay=0.01,
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     load_best_model_at_end=True,
#     push_to_hub=True,                                  # <-- Set to True to save to Hugging Face
#     hub_model_id="GouravBanerjee/DocOrNoDoc",        # <-- REPLACE WITH YOUR HUGGING FACE USERNAME
#     report_to="none"
# )
# 
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=processed_dataset["train"],
#     eval_dataset=processed_dataset["test"],
#     processing_class=processor,
#     compute_metrics=compute_metrics,
# )
# 
# print("🚀 Training has officially started! Watching for accuracy...")
# trainer.train()
# 
# # After training finishes, we explicitly push the processor and the model to the hub
# trainer.push_to_hub()
# processor.push_to_hub("GouravBanerjee/DocOrNoDoc")


"""Testing"""
print("\n--- 🧪 Testing ---")
print("⏳ Finalizing environment...")

reader = easyocr.Reader(['en'])
checkpoints = glob.glob("./document_classifier_model/checkpoint-*")

# Hugging Face fallback configuration
HF_MODEL_ID = "GouravBanerjee/DocOrNoDoc" # <-- REPLACE WITH YOUR HUGGING FACE USERNAME

if checkpoints:
    best_model_path = max(checkpoints, key=os.path.getctime)
    print(f"🧠 Loading Brain from LOCAL: {best_model_path}")
    model = LayoutLMv3ForSequenceClassification.from_pretrained(best_model_path)
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
else:
    print(f"☁️ Local model not found. Downloading from Hugging Face: {HF_MODEL_ID}...")
    try:
        model = LayoutLMv3ForSequenceClassification.from_pretrained(HF_MODEL_ID)
        processor = LayoutLMv3Processor.from_pretrained(HF_MODEL_ID, apply_ocr=False)
        print("✅ Successfully loaded model and processor from Hugging Face!")
    except Exception as e:
        print(f"❌ Could not load from Hugging Face. Have you trained/pushed the model yet? Error: {e}")
        exit(1)

def get_ocr_data(img):
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

test_folder = "Dataset/Test"
if not os.path.exists(test_folder):
    os.makedirs(test_folder)
    print(f"Created '{test_folder}/' directory. Please add files to it for testing.")

final_results = []
print(f"📂 Scanning folder: {test_folder}...\n")

for filename in sorted(os.listdir(test_folder)):
    file_path = os.path.join(test_folder, filename)
    if filename.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg', '.tif')):
        try:
            if filename.lower().endswith(".pdf"):
                img = convert_from_path(file_path, first_page=1, last_page=1, dpi=200)[0].convert("RGB")
            else:
                img = Image.open(file_path).convert("RGB")

            words, boxes = get_ocr_data(img)
            if not words: continue

            inputs = processor(
                img,
                text=words,
                boxes=boxes,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="max_length"
            )

            with torch.no_grad():
                outputs = model(**inputs)

            prediction = torch.argmax(outputs.logits, dim=-1).item()
            probs = torch.softmax(outputs.logits, dim=-1)
            confidence = probs[0][prediction].item()

            label = "ACCOUNTING (BUSY)" if prediction == 1 else "OTHER/NON-ACCOUNTING"
            final_results.append({"file": filename, "label": label, "conf": confidence})

            icon = "✅" if prediction == 1 else "📄"
            print(f"{icon} {filename:35} -> {label} ({confidence*100:.1f}%)")

        except Exception as e:
            print(f"❌ Error in {filename}: {e}")

print("\n" + "="*50 + "\n📊 BATCH TEST SUMMARY\n" + "="*50)
print(f"Total Files Scanned: {len(final_results)}")
print(f"Found Accounting:    {sum(1 for r in final_results if r['label'] == 'ACCOUNTING (BUSY)')}")
print("="*50)