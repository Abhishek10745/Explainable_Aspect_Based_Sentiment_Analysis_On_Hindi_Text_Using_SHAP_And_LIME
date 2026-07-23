# 🧠 Explainable Aspect-Based Sentiment Analysis for Hindi using MuRIL, SHAP & LIME

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Model](https://img.shields.io/badge/Model-MuRIL-success)
![Task](https://img.shields.io/badge/Task-ABSA-brightgreen)
![Language](https://img.shields.io/badge/Language-Hindi-orange)
![License](https://img.shields.io/badge/License-MIT-blue)

</p>

---

## 📌 Overview

This project implements an **Explainable Aspect-Based Sentiment Analysis (ABSA)** pipeline for **Hindi text** using Google's **MuRIL Transformer**.

Unlike conventional sentiment analysis, the model predicts the sentiment **towards a specific aspect** inside a sentence.

To improve transparency, predictions are explained using

- 🟢 SHAP (SHapley Additive Explanations)
- 🔵 LIME (Local Interpretable Model-Agnostic Explanations)

making every prediction interpretable at the **token level**.

---

# ✨ Features

| Feature | Description |
|----------|-------------|
| 🇮🇳 Hindi Support | Uses Google's MuRIL multilingual transformer |
| 🎯 Aspect Highlighting | Marks aspect using `[ASP] ... [ASP]` |
| ⚖️ Weighted Loss | Handles class imbalance |
| 📈 Fine-tuning | HuggingFace Trainer |
| 🔍 Explainability | SHAP + LIME |
| 📊 Evaluation | Accuracy, Precision, Recall, F1 |
| ⚡ GPU Training | CUDA supported |

---

# 🏗 Project Pipeline

```text
                 Hindi Review
                       │
                       ▼
            Text Cleaning & Normalization
                       │
                       ▼
             Aspect Highlighting
        [ASP] कैमरा [ASP]
                       │
                       ▼
           MuRIL Tokenizer
                       │
                       ▼
          MuRIL Transformer Encoder
                       │
                       ▼
      Sentiment Classification Head
                       │
                       ▼
      Negative / Neutral / Positive
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
       SHAP                     LIME
          ▼                         ▼
 Token Importance          Local Explanation
```

---

# 📂 Dataset

The dataset is downloaded automatically.

```
combined_hindi_absa.csv
```

### Dataset Split

| Split | Percentage |
|--------|-----------:|
| Train | 80% |
| Validation | 10% |
| Test | 10% |

---

# 🛠 Installation

```bash
pip install torch torchvision transformers datasets accelerate shap lime pandas gdown scikit-learn
```

---

# 🧹 Preprocessing

✔ Remove URLs

✔ Remove HTML Tags

✔ Unicode normalization

✔ Remove unnecessary punctuation

✔ Aspect highlighting

Example

Input

```
इस मोबाइल का कैमरा शानदार है
```

↓

```
इस मोबाइल का [ASP] कैमरा [ASP] शानदार है
```

---

# 🧠 Model Configuration

| Parameter | Value |
|-----------|------:|
| Base Model | google/muril-base-cased |
| Epochs | 10 |
| Batch Size | 32 |
| Learning Rate | 1e-5 |
| Max Length | 128 |
| Warmup Steps | 137 |
| Loss | Weighted Cross Entropy |

---

# 📊 Evaluation Metrics

The model is evaluated using

- Accuracy
- Precision
- Recall
- Weighted F1-score

Example

| Metric | Score |
|---------|------:|
| Accuracy | XX.XX |
| Precision | XX.XX |
| Recall | XX.XX |
| F1-score | XX.XX |

---

# 🔍 Explainability

## LIME

LIME builds a **local surrogate linear model** around the prediction.

Workflow

```text
Original Sentence
        │
        ▼
Generate Perturbed Samples
        │
        ▼
Model Predictions
        │
        ▼
Distance Weighting
        │
        ▼
Weighted Linear Regression
        │
        ▼
Word Importance
```

---

## SHAP

SHAP computes **Shapley values** from cooperative game theory.

Workflow

```text
Original Sentence
        │
        ▼
Generate Word Subsets
        │
        ▼
Predict Every Combination
        │
        ▼
Marginal Contribution
        │
        ▼
Shapley Values
```

---

# 📈 Example Explanation

Sentence

```
इस मोबाइल का कैमरा शानदार है
```

Aspect

```
कैमरा
```

Prediction

```
Positive (0.96)
```

### LIME

```
+ शानदार
+ कैमरा
- मोबाइल
```

---

### SHAP

```
████████ शानदार

██████ कैमरा

██ मोबाइल
```

---

# 📁 Project Structure

```text
Hindi-ABSA/
│
├── data/
│   └── combined_hindi_absa.csv
│
├── notebooks/
│
├── models/
│
├── utils/
│
├── train.py
├── predict.py
├── explain.py
├── requirements.txt
└── README.md
```

---

# 🚀 Usage

## Train

```bash
python train.py
```

---

## Predict

```bash
python predict.py
```

---

## Explain using LIME

```python
from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(
    class_names=['Negative','Neutral','Positive'],
    split_expression=r'\s+'
)

exp = explainer.explain_instance(
    text_to_explain,
    lime_predictor,
    num_features=10,
    num_samples=500
)

exp.show_in_notebook(text=True)
```

---

## Explain using SHAP

```python
explainer = shap.Explainer(model_predict)

shap_values = explainer([text])

shap.plots.text(shap_values[0])
```

---

# 📌 Future Improvements

- 🔹 Attention visualization
- 🔹 Integrated Gradients
- 🔹 Grad-CAM for Transformers
- 🔹 Multilingual ABSA
- 🔹 Web application deployment
- 🔹 ONNX optimization

---

# 📚 Citation

If you use this repository in your work, please cite it.

---

# ⭐ Acknowledgements

- Google MuRIL
- HuggingFace Transformers
- SHAP
- LIME
- Scikit-learn
