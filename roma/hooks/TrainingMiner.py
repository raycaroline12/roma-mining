import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from unidecode import unidecode
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

stopwords = nltk.corpus.stopwords.words('portuguese')
        
special_characters = ['!', '-', '?', '(', ')', ':', '.', ',', '@', "#",
                        '*', '&', '%', '$', "'", '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                        '"', '[', ']', '{', '}', '´', '`', '+']
stemmer = SnowballStemmer('portuguese')

class Training_Miner:
    
    def __init__(self):
        self.columns_topic = ['COMENTÁRIO', 'CLASSIFICAÇÃO']
        self.columns_sentiment = ['COMENTÁRIO', 'SENTIMENTO']
    
    def get_topics_dataset(self, dataset):
        topics_ds = dataset[self.columns_topic]
        
        return topics_ds
    
    def get_sentiment_dataset(self, dataset):
        sentiment_ds = dataset[self.columns_sentiment]
        
        return sentiment_ds
    
    def pre_processing(self, dataset):
        dataset['COMENTÁRIO'] = dataset['COMENTÁRIO'].str.lower()
        dataset['COMENTÁRIO'] = dataset.COMENTÁRIO.apply(unidecode)
        dataset['COMENTÁRIO'] = dataset.apply(lambda row: nltk.word_tokenize(row['COMENTÁRIO']), axis=1)
        dataset['COMENTÁRIO'] = dataset['COMENTÁRIO'].apply(lambda x: [item for item in x if item not in special_characters])
        dataset['COMENTÁRIO'] = dataset['COMENTÁRIO'].apply(lambda x: [item for item in x if item not in stopwords])
        dataset['COMENTÁRIO'] = dataset['COMENTÁRIO'].apply(lambda x: [stemmer.stem(y) for y in x])
        dataset['COMENTÁRIO'] = dataset['COMENTÁRIO'].apply(lambda x: " ".join(x))

        return dataset
    
    def bag_of_words(self, dataset):
        bow = CountVectorizer(ngram_range=(1,3), stop_words=stopwords)
        bow.fit(dataset.COMENTÁRIO)
        text_bow = bow.transform(dataset.COMENTÁRIO)
        
        return text_bow
    
    def bow_new_data(self, dataset1, dataset2):
        bow = CountVectorizer(ngram_range=(1,3), stop_words=stopwords)
        bow.fit(dataset1.COMENTÁRIO)
        text_bow = bow.transform(dataset2.COMENTÁRIO)
        
        return text_bow
        
    def split_train_test_sentiment(self, dataset, text_bow):
        X_trainUCV, X_testUCV, y_trainUCV, y_testUCV = train_test_split(
            text_bow, dataset["SENTIMENTO"], test_size = 0.3, random_state = 1)
        return X_trainUCV, X_testUCV, y_trainUCV, y_testUCV

    def split_train_test_topics(self, dataset, text_bow):
        X_trainUCV, X_testUCV, y_trainUCV, y_testUCV = train_test_split(
            text_bow, dataset["CLASSIFICAÇÃO"], test_size = 0.3, random_state = 1)
        return X_trainUCV, X_testUCV, y_trainUCV, y_testUCV
    
    def train_multinomial_nb(self, X_train, y_train):
        mnb = MultinomialNB().fit(X_train, y_train)
        
        return mnb
    
    def train_complement_nb(self, X_train, y_train):
        cnb = ComplementNB().fit(X_train, y_train)
        
        return cnb
    
    def predict(self, trained_nb, X_test):
        y_prediction = trained_nb.predict(X_test)
        
        return y_prediction
    
    def evaluate_sentiment(self, y_prediction, y_test):       
        accuracy = accuracy_score(y_test, y_prediction)
        recall = recall_score(y_test, y_prediction)
        precision = precision_score(y_test, y_prediction)
        f1 = f1_score(y_test, y_prediction)
        
        evaluation = [accuracy, recall, precision, f1_score]
        
        return evaluation

    
    def evaluate_topics(self, y_prediction, y_test):        
        accuracy = accuracy_score(y_test, y_prediction)
        recall = recall_score(y_test, y_prediction, average='weighted')
        precision = precision_score(y_test, y_prediction, average='weighted')
        f1 = f1_score(y_test, y_prediction, average='weighted')
        
        evaluation = [accuracy, recall, precision, f1_score]
        
        return evaluation
    
    def evaluation_report(self,y_prediction, y_test):        
        report = classification_report(y_test, y_prediction)
        
        return report
