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

def main():
    # -------------------------------------------------------------------------
    # 1. Download necessary NLTK data (stopwords) if not already available
    # -------------------------------------------------------------------------
    nltk.download('stopwords', quiet=True)
    
    # Print current working directory to debug file path issues
    print(f"Current working directory: {os.getcwd()}")
    
    # -------------------------------------------------------------------------
    # 2. Importing the dataset
    #    Make sure 'Restaurant_Reviews.tsv' is in the same directory or specify 
    #    the full path if it's located elsewhere.
    # -------------------------------------------------------------------------
    dataset_path = 'Restaurant_Reviews.tsv'  # Adjust if needed
    dataset = pd.read_csv(dataset_path, delimiter='\t', quoting=3)
    
    # Show some basic info about the dataset
    print("\n--- Dataset Head ---")
    print(dataset.head())
    print("\n--- Dataset Info ---")
    print(dataset.info())
    
    # -------------------------------------------------------------------------
    # 2a. Visualize distribution of target variable (assuming last column is 'Liked')
    # -------------------------------------------------------------------------
    # If your dataset's column name is something else, adjust below:
    target_col = dataset.columns[-1]
    
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_col, data=dataset, palette='viridis')
    plt.title("Distribution of Liked vs. Not Liked Reviews")
    plt.xlabel("Review Liked? (0=No, 1=Yes)")
    plt.ylabel("Count")
    plt.show()
    
    # -------------------------------------------------------------------------
    # 3. Cleaning the texts
    # -------------------------------------------------------------------------
    corpus = []
    for i in range(len(dataset)):
        # a) Keep only letters
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        
        # b) Convert to lower case
        review = review.lower()
        
        # c) Split into individual words (tokenize)
        review = review.split()
        
        # d) Initialize Porter Stemmer
        ps = PorterStemmer()
        
        # e) Load stopwords, remove 'not' if present
        all_stopwords = stopwords.words('english')
        if 'not' in all_stopwords:
            all_stopwords.remove('not')
        
        # f) Stemming and stopword removal
        review = [ps.stem(word) for word in review if word not in all_stopwords]
        
        # g) Rejoin the cleaned words
        review = ' '.join(review)
        
        # h) Append to corpus
        corpus.append(review)

    print(f"\n--- First 5 cleaned reviews ---\n{corpus[:5]}\n")
    
    # -------------------------------------------------------------------------
    # 4. Creating the Bag of Words model
    # -------------------------------------------------------------------------
    cv = CountVectorizer(max_features=1500)
    X = cv.fit_transform(corpus).toarray()
    
    # Assuming the last column in the dataset is the sentiment (Liked: 0 or 1)
    y = dataset.iloc[:, -1].values
    
    # -------------------------------------------------------------------------
    # 5. Splitting the dataset into the Training set and Test set
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0
    )
    
    # -------------------------------------------------------------------------
    # 6. Training the Naive Bayes model on the Training set
    # -------------------------------------------------------------------------
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    
    # -------------------------------------------------------------------------
    # 7. Predicting the Test set results
    # -------------------------------------------------------------------------
    y_pred = classifier.predict(X_test)
    
    print("--- Predicted vs Actual for Test Set ---")
    comparison = np.concatenate(
        (y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 
        1
    )
    print(comparison)
    
    # -------------------------------------------------------------------------
    # 8. Making the Confusion Matrix and Checking Accuracy
    # -------------------------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\nConfusion Matrix:")
    print(cm)
    print(f"\nAccuracy Score: {accuracy:.2f}")
    
    # -------------------------------------------------------------------------
    # 8a. Visualize the Confusion Matrix as a heatmap
    # -------------------------------------------------------------------------
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

if __name__ == "__main__":
    main()
