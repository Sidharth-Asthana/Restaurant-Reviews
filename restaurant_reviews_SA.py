import os
import re
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from wordcloud import WordCloud


def main():
    # -------------------------------------------------------------------------
    # 1. NLTK and File Setup
    # -------------------------------------------------------------------------
    nltk.download('stopwords', quiet=True)
    
    print(f"Current working directory: {os.getcwd()}")
    dataset_path = 'Restaurant_Reviews.tsv'  # Adjust if needed
    
    # -------------------------------------------------------------------------
    # 2. Load the Dataset
    # -------------------------------------------------------------------------
    dataset = pd.read_csv(dataset_path, delimiter='\t', quoting=3)
    print("\n--- Dataset Head ---")
    print(dataset.head())

    # -------------------------------------------------------------------------
    # 3. Basic Visualization of Target
    # -------------------------------------------------------------------------
    target_col = dataset.columns[-1]  # The last column should be the sentiment
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_col, data=dataset, palette='viridis')
    plt.title("Distribution of Positive (1) vs. Negative (0) Reviews")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.show()
    
    # -------------------------------------------------------------------------
    # 4. Text Cleaning
    # -------------------------------------------------------------------------
    corpus = []
    for i in range(len(dataset)):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])  # Keep only letters
        review = review.lower()                                  # Lowercase
        review = review.split()                                  # Tokenize
        ps = PorterStemmer()
        
        # Load stopwords (and remove "not" if present)
        all_stopwords = stopwords.words('english')
        if 'not' in all_stopwords:
            all_stopwords.remove('not')
        
        # Stemming and stopword removal
        review = [ps.stem(word) for word in review if word not in all_stopwords]
        review = ' '.join(review)
        corpus.append(review)

    print("\n--- First 5 Cleaned Reviews ---")
    print(corpus[:5])
    
    # -------------------------------------------------------------------------
    # 5. Bag of Words Model
    # -------------------------------------------------------------------------
    cv = CountVectorizer(max_features=1500)
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values  # 0 or 1
    
    # -------------------------------------------------------------------------
    # 6. Train-Test Split
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0
    )
    
    # -------------------------------------------------------------------------
    # 7. Train a Naive Bayes Classifier
    # -------------------------------------------------------------------------
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    
    # -------------------------------------------------------------------------
    # 8. Predictions and Performance
    # -------------------------------------------------------------------------
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n--- Confusion Matrix ---")
    print(cm)
    print(f"\nAccuracy Score: {accuracy:.2f}\n")
    
    # Plot Confusion Matrix as a Heatmap
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    
    # -------------------------------------------------------------------------
    # 9. Word Clouds for Positive vs. Negative Reviews
    # -------------------------------------------------------------------------
    # Separate reviews based on actual sentiment
    positive_reviews = [corpus[i] for i in range(len(corpus)) if y[i] == 1]
    negative_reviews = [corpus[i] for i in range(len(corpus)) if y[i] == 0]
    
    # Combine all positive reviews into one string
    positive_text = " ".join(positive_reviews)
    negative_text = " ".join(negative_reviews)

    # Create word clouds
    wordcloud_positive = WordCloud(width=600, height=400, background_color='white',
                                   colormap='Greens').generate(positive_text)
    wordcloud_negative = WordCloud(width=600, height=400, background_color='white',
                                   colormap='Reds').generate(negative_text)
    
    # Plot them side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(wordcloud_positive, interpolation='bilinear')
    axes[0].axis('off')
    axes[0].set_title("Word Cloud: Positive Reviews")

    axes[1].imshow(wordcloud_negative, interpolation='bilinear')
    axes[1].axis('off')
    axes[1].set_title("Word Cloud: Negative Reviews")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
