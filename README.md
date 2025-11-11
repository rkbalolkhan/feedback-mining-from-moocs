# 🎓 Feedback Mining from MOOCs

This project applies **Machine Learning (ML)** and **Natural Language Processing (NLP)** to analyze student feedback from **Massive Open Online Courses (MOOCs)** and extract meaningful sentiment insights.

---

## 📘 Features

- 🧠 Sentiment Analysis (Positive / Negative / Neutral)  
- 🔤 TF-IDF Feature Extraction  
- 🤖 Logistic Regression & Naive Bayes Models  
- 📊 Visualizations: Word Clouds, Sentiment Distributions  

---

## 🧩 Directory Structure

```
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
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/rkbalolkhan/feedback-mining-from-moocs.git
cd feedback-mining-from-moocs
```

### 2️⃣ (Optional) Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # For Mac/Linux
venv\Scripts\activate      # For Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Add Dataset
Place your dataset (e.g. `coursera_reviews.csv`) inside:
```
data/raw/
```

---

## 🧠 About the Project

This repository contains the complete workflow for **Feedback Mining from MOOCs**, which involves:
- Data preprocessing (cleaning, tokenization, lemmatization)
- Feature extraction using TF-IDF
- Model training using Logistic Regression & Naive Bayes
- Performance evaluation & visualization

The goal is to extract meaningful **sentiment insights** from learner feedback to improve course quality and instructor performance.

---

## 🧩 Future Work

- Implement **Aspect-Based Sentiment Analysis (ABSA)**  
- Integrate **Transformer Models** (BERT, RoBERTa)  
- Deploy as an interactive **Streamlit web app**  
- Add **Topic Modeling** (LDA) for feedback clustering  

---

## 👨‍💻 Author

**Rahematullah Balolkhan**  
B.Tech CSE — Lovely Professional University  
Full Stack Developer | Machine Learning Enthusiast  

🌐 [LinkedIn](https://www.linkedin.com/in/rkbalolkhan/)  
📸 [Instagram](https://www.instagram.com/rk.balolkhan)

---

## 🪪 License

This project is licensed under the **MIT License** — you’re free to use, modify, and distribute it with proper attribution.

---

> _"Data becomes knowledge only when it speaks — this project lets learner feedback speak clearly."_ 💡
