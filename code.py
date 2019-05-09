import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
import sys
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix

def train():
    nltk.download('stopwords')
    en_stops = set(stopwords.words('english'))
    english_words = set(nltk.corpus.words.words())
    txt_file_names = glob.glob("emaildataset/**/*.txt")
    X = []
    Y = []
    for name in txt_file_names:
        f = open(name, "r")
        contents = f.read()
        temp_contents = []
        words = contents.split()
        for word in words:
            # remove stop words non english words and single character words
            if word not in en_stops and word in english_words and len(word) > 1:
                temp_contents.append(word)
        contents = ' '.join(temp_contents)
        X.append(contents)
        if 'spmsg' in name:
            Y.append('spam')
        else:
            Y.append('ham')

    # have a min_df of .0001% which means that I want to ignore terms that are less than threshold given
    # have max_df of 80% which mean that I want to ignore terms that are found in 80% of the document
    vectorizer = CountVectorizer(max_features=1000, min_df=.0001, max_df=.8)
    train_data_features = vectorizer.fit_transform(X)
    joblib.dump(vectorizer,'saved_vectorizer.pkl')

    clf = SVC(kernel='sigmoid')
    clf.fit(train_data_features, Y)
    # Output a pickle file for the model
    joblib.dump(clf, 'saved_model.pkl')


def predict():
    # load the models
    vectorizer = joblib.load('saved_vectorizer.pkl')
    clf = joblib.load('saved_model.pkl')
    # Get txt file name from given user using sys module
    predict_file_name = sys.argv[1]
    f = open(predict_file_name, "r")
    pred_txt = f.read()
    pred_txt = [pred_txt]
    pred_txt = vectorizer.transform(pred_txt)
    pred_txt = pred_txt.toarray()
    y_pred = clf.predict(pred_txt)
    result = y_pred[-1]
    if result == 'spam':
        print(1)
    if result == 'ham':
        print(0)

def main():
    # train()
    predict()

main()
