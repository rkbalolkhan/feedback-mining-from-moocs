# Feedback Mining from MOOCs 🎓

This project applies Machine Learning and Natural Language Processing (NLP) to analyze student feedback from Massive Open Online Courses (MOOCs) and extract meaningful sentiment insights.

## 📘 Features
- Sentiment analysis (positive/negative/neutral)
- TF-IDF feature extraction
- Logistic Regression & Naive Bayes models
- Visualizations: Word clouds, sentiment distributions

## 🧩 Directory Structure

feedback-mining-from-moocs/
│
├── data/
│   ├── raw/
│   │   └── coursera_reviews.csv          # Original unmodified dataset
│   ├── processed/
│   │   └── cleaned_reviews.csv           # After cleaning, lemmatization, etc.
│   └── external/                         # Any additional datasets or test sets
│
├── notebooks/
│   ├── 01_data_exploration.ipynb         # EDA & visualization
│   ├── 02_baseline_sentiment_model.ipynb # TF-IDF + Logistic Regression/Naive Bayes
│   ├── 03_model_optimization.ipynb       # Hyperparameter tuning, embeddings, etc.
│   └── 04_bert_experiments.ipynb         # Transformer-based sentiment model
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py             # Data cleaning functions
│   ├── feature_extraction.py             # TF-IDF, embeddings, etc.
│   ├── train_model.py                    # Model training scripts
│   ├── evaluate_model.py                 # Evaluation metrics, confusion matrix
│   └── visualization.py                  # Word clouds, plots, etc.
│
├── models/
│   ├── logistic_regression.pkl           # Saved sklearn model
│   ├── vectorizer.pkl                    # Saved TF-IDF vectorizer
│   └── bert_model/                       # Folder for Hugging Face fine-tuned model
│
├── reports/
│   ├── figures/                          # Word clouds, sentiment distribution plots
│   └── research_paper_draft.docx         # Your report or thesis
│
├── requirements.txt                      # Python dependencies
├── README.md                             # Project overview and setup guide
└── .gitignore                            # Ignore data/models when using Git
