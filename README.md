Explainable Aspect-Based Sentiment Analysis on Hindi Text using SHAP and LIME
📌 Project Overview
This repository provides a complete pipeline for performing Aspect-Based Sentiment Analysis (ABSA) on Hindi text. It utilizes the pre-trained google/muril-base-cased model to classify the sentiment of specific aspects within a sentence into three polarities: Negative (0), Neutral (1), and Positive (2). To ensure model transparency, the project integrates LIME (Local Interpretable Model-Agnostic Explanations) and SHAP (SHapley Additive exPlanations) to explain the model's predictions at the token level.

✨ Key Features
Hindi Text Preprocessing: Automated cleaning of Hindi text, including the removal of URLs, HTML tags, and unnecessary punctuation, alongside unicode normalization.

Targeted Aspect Highlighting: Highlights specific aspects within a sentence using [ASP] token tags (e.g., [ASP] कैमरा [ASP]) to give the model context.

Class Imbalance Handling: Implements a custom WeightedLossTrainer leveraging compute_class_weight to apply balanced weights via a weighted Cross-Entropy Loss function.

Explainable AI (XAI): Uses SHAP and LIME text explainers to visualize how individual words contribute to the positive, negative, or neutral sentiment of the targeted aspect.

🛠️ Dependencies
Ensure you have the following libraries installed before running the project:

Bash
pip install torch torchvision transformers datasets accelerate shap lime pandas gdown scikit-learn
📂 Dataset
The project automatically downloads the combined_hindi_absa.csv dataset via gdown.

The dataset is split into 80% Training, 10% Validation, and 10% Testing subsets.

Labels are encoded using LabelEncoder to map text polarities to numeric values.

🧠 Model Training Details
Base Model: google/muril-base-cased

Max Sequence Length: 128 tokens

Epochs: 10

Batch Size: 32 (for both training and evaluation)

Learning Rate: 1e-5 with a linear scheduler and 137 warmup steps

Evaluation Metrics: Accuracy, Precision, Recall, and F1-Score (Weighted)

🔍 Explainability (XAI) Walkthrough
This project includes in-depth mathematical walkthroughs for both explainers:

LIME (Local Interpretable Model-Agnostic Explanations)
LIME creates a local surrogate model by generating a perturbed dataset around the target prediction.

Perturbation: The text is broken into tokens, and a binary matrix is created representing different combinations of the presence/absence of those tokens.

Distance & Weighting: The Hamming distance from the original sentence is calculated, and an exponential kernel assigns higher weights to samples closer to the original instance.

Regression: A weighted linear regression model is solved to find feature contributions, explaining the base probability and which words act as positive/negative drivers.

SHAP (SHapley Additive exPlanations)
SHAP calculates the marginal contribution of each text token across all possible subsets of words.

Subset Predictions: The model is queried for every possible combination of words in the sentence.

Shapley Weights: Weights are calculated based on the subset size using combinatorial math.

Final Contribution: The individual token contributions are aggregated to show exactly how much a specific token shifts the prediction away from the neutral base value.

🚀 Usage
1. Prepare the Data & Train the Model:
Run the training pipeline to clean the text, format the inputs with [ASP] tags, and fine-tune the MuRIL model.

2. Test & Predict:
The Trainer will predict outputs for the test dataset and return metrics like Accuracy, F1, Precision, and Recall.

3. Generate Explanations:
Pass the fine-tuned model and tokenizer to the SHAP and LIME explainers to visualize predictions.

Python
# Example for LIME
from lime.lime_text import LimeTextExplainer

explainer_lime = LimeTextExplainer(class_names=['Negative', 'Neutral', 'Positive'], split_expression=r'\s+')
lime_exp = explainer_lime.explain_instance(text_to_explain, lime_predictor, num_features=10, num_samples=500)
lime_exp.show_in_notebook(text=True)
