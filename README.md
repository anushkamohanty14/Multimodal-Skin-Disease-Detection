# Multimodal Skin Lesion Classification (HAM10000)

## Dataset Selection

The **HAM10000 dataset** was selected for its **multimodal structure** , dermoscopic images with textual metadata (age, sex, localization, diagnosis), and its moderate size (~10,000 samples), making it suitable for the compute and timely submission constraints. **Data alignment** was ensured via the image_id key for text–image fusion.

---

## Overview

This project fine-tunes two pretrained transformer models , **PubMedBERT** for clinical metadata and **Swin Transformer** for dermoscopic images, and fuses their embeddings for skin lesion classification on the **HAM10000** dataset.
The pipeline integrates **natural language (metadata)** and **visual (image)** modalities through a **gated fusion neural network**, achieving balanced diagnostic performance.

---

## File Structure

```
Anushka_MultimodalDiseaseDetection/
├── Anushka_MultimodalDiseaseDetection.ipynb     # single notebook with full pipeline
│
├── artifacts/
│   ├── metadata_aligned.csv
│   ├── pubmedbert_ham10000_refined/
│   ├── swin_ham10000_refined/
│   ├── pubmedbert_ham10000_emb.npy
│   ├── swin_ham10000_emb.npy
│   ├── ham10000_fused_emb.npy
│   ├── fusion_classifier_final.keras
│   ├── fusion_best_by_macrof1.weights.h5
│   ├── fusion_X_train.npy
│   ├── fusion_y_train.npy
│   ├── label_classes.txt
│   └── ham10000_ids.npy
│
├── outputs/
│   ├── training_curves.png
│   └── confusion_matrix.png
│   
│  
│
├── dataset/
│   ├── HAM10000_metadata.csv
│   ├── HAM10000_images_part_1/
│   └── HAM10000_images_part_2/
│
└── README.md

```

---

## Pipeline Summary

### Step 1: Data Alignment

* Combined `HAM10000_metadata.csv` with both image folders.
* Merged using `image_id` for one-to-one pairing.
* Output: `artifacts/metadata_aligned.csv`

### Step 2: Text Modality — PubMedBERT

* Model: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`
* Text fields: `lesion_id`, `age`, `sex`, `localization`, `dx_type`
* Preprocessing: lowercasing, punctuation and digit removal, whitespace normalization
* Fine-tuned using Hugging Face Trainer (2 epochs, batch size 16)
* Output:

  ```
  artifacts/pubmedbert_ham10000_refined/
  artifacts/pubmedbert_ham10000_emb.npy
  ```

### Step 3: Image Modality — Swin Transformer

* Model: `microsoft/swin-tiny-patch4-window7-224`
* Frozen all but final block and classifier
* Optimizer: AdamW (LR = 5e-5, epochs = 3)
* Output:

  ```
  artifacts/swin_ham10000_refined/
  artifacts/swin_ham10000_emb.npy
  ```

### Step 4: Embedding Fusion

* Combined normalized embeddings from both modalities
* Gated fusion mechanism:

  ```
  gate_text = sigmoid(Dense(text_dim))
  gate_image = sigmoid(Dense(image_dim))
  fused = concat([text * gate_text, image * gate_image])
  ```
* Output: `artifacts/ham10000_fused_emb.npy`

### Step 5: Fusion Classifier

* Architecture: 512 → 256 → 128 dense layers + dropout + L2 regularization
* Loss: Sparse Focal Loss (γ=2.0, α=0.75)
* Split: Train/Val/Test = 70/15/15 (grouped by lesion_id)
* Early stopping on validation macro-F1
* Outputs:

  ```
  fusion_classifier_final.keras
  fusion_best_by_macrof1.weights.h5
  ```

---

## Evaluation Summary

The multimodal fusion model was trained and evaluated across **three independent runs** to ensure consistency. All evaluations used the same train/validation/test split and macro-averaged metrics.

|  Run  | Accuracy | Precision | Recall | F1-score |
| :---: | :------: | :-------: | :----: | :------: |
| **1** |  0.9054  |   0.8372  | 0.8124 |  0.8170  |
| **2** |  0.9108  |   0.8225  | 0.7816 |  0.7948  |
| **3** |  0.9144  |   0.8651  | 0.7827 |  0.8086  |


Average Performance (across runs):

** Accuracy: 0.9102

** Precision: 0.8416

** Recall: 0.7922

** F1-score: 0.8068

These results demonstrate stable and balanced multimodal learning, with precision and recall in close range, ensuring both low false positives and low false negatives, essential for clinical reliability.

---

## Modality Contribution (Ablation)

| Modality                       |  Macro F1 |  Accuracy |
| :----------------------------- | :-------: | :-------: |
| **Image-only (Swin)**          |   0.750   |   0.880   |
| **Text-only (PubMedBERT)**     |   0.720   |   0.840   |
| **Fusion (Swin + PubMedBERT)** | **0.817** | **0.905** |


The fusion model outperformed unimodal baselines, confirming complementary feature learning.

---


## Data Provenance

**HAM10000 — with 10000 Training Images**
Tschandl et al., 2018 — Harvard Dataverse
[https://doi.org/10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T)

---

## Author

**Anushka Mohanty**
B.Tech CSE : 3rd Year
Mahindra University


> *Multimodal deep learning for skin lesion classification using PubMedBERT and Swin Transformer fusion.*
