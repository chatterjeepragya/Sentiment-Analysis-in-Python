"""
The following example is to categorize a movie review to positive or negative
Reimplemented from http://www.nltk.org/book/ch06.html
"""
import nltk
from nltk.corpus import movie_reviews

# STEP 1: OBTAIN DATA SET
# obtain a list of tuples with words in movie reviews and their categories (words, category)
# movie_reviews is a corpus of 2k movie reviews with sentiment polarity labels
# more information about nltk corpus read chapter 2 https://www.nltk.org/book/ch02.html
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# STEP 2: DEFINE FEATURE EXTRACTOR
# define features and ways to extract the features
# We only check the top 2000 most frequent words to speed up the feature extraction
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
top_words = list(all_words)[:2000]  # obtain top 2000 most frequent words


# define a function word_features that simply checks whether a word is present in a give document or not.
# the input parameter document is a list of tokens in a document
# the returned variable features is a dictionary of {feature: value} pairs
def word_features(document):
    doc_words = set(document)  # get the word set in the document
    features = {}  # define a dictionary features{} to store feature name: value pairs
    for word in top_words:
        features['contains({})'.format(word)] = (word in doc_words)
    return features


# STEP 3: TRAINING
# for each document and category in documents, extract the features of the data
# featuresets is a list of tuples, each tuple has two elements: 1. word_features(d), 2. category
featuresets = [(word_features(d), c) for (d, c) in documents]
# split the data into training and testing, the first 100 as testing and the rest for training
train_set, test_set = featuresets[100:], featuresets[:100]
# train the data and use the NaiveBayes Classifier
classifier = nltk.NaiveBayesClassifier.train(train_set)

# STEP 4: PREDICTING
# use the trained classifier to predict the labels of testing set
for (features, category) in test_set:
    predicted = classifier.classify(features)
    print("predicted: " + predicted, "true: " + category)
print(nltk.classify.accuracy(classifier, test_set))  # print out the accuracy of the prediction
print(classifier.show_most_informative_features(5))  # print out the most informative features
