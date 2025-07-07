# Sentiment Analysis on IMDb Movie Reviews

This project performs sentiment analysis on the IMDb movie reviews dataset using Python and machine learning techniques. The main script, `sentiment_analysis_imdb.py`, loads, preprocesses, and classifies movie reviews as positive or negative using Logistic Regression.

## Features
- Loads IMDb dataset from local folders (positive and negative reviews)
- Text preprocessing: lowercasing, stopword removal, stemming
- Feature extraction using TF-IDF vectorization
- Label encoding for sentiment classes
- Model training and evaluation using Logistic Regression
- Plots and saves confusion matrix and precision/recall/F1-score bar charts

## Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn
- nltk
- matplotlib
- seaborn

Install dependencies with:
```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn
```

## Dataset
Download the [IMDb Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) and extract it. Update the script if your dataset path differs from `./aclImdb/train` and `./aclImdb/test`.

## Usage
1. Ensure the dataset is extracted and the directory structure matches the script's expectations.
2. Run the script:
   ```bash
   python sentiment_analysis_imdb.py
   ```
3. The script will output model metrics and save plots:
   - `confusion_matrix_lr.png`: Confusion matrix for Logistic Regression
   - `precision_recall_f1_lr.png`: Precision, Recall, and F1-Score bar chart

## Output
- Console output: Accuracy, F1 Score, classification report
- Images: Confusion matrix and metrics bar chart saved in the project directory

