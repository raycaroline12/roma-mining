from hooks.TrainingMiner import Training_Miner
import pandas as pd
from itertools import chain


def RunTraining():
    reviews_topic = pd.read_csv('hooks/data/training_set_topics.csv')
    reviews_topic = pd.DataFrame(reviews_topic)
    reviews_sentiment = pd.read_csv('hooks/data/training_set_sentiment.csv')
    reviews_sentiment = pd.DataFrame(reviews_sentiment)

    reviews_topic = Training_Miner().get_topics_dataset(reviews_topic)
    reviews_sentiment = Training_Miner().get_sentiment_dataset(reviews_sentiment)

    reviews_sentiment = Training_Miner().pre_processing(reviews_sentiment)
    reviews_topics = Training_Miner().pre_processing(reviews_topic)

    bow_sentiment = Training_Miner().bag_of_words(reviews_sentiment)
    bow_topics = Training_Miner().bag_of_words(reviews_topics)

    X_train_s, X_test_s, y_train_s, y_test_s = Training_Miner().split_train_test_sentiment(reviews_sentiment, bow_sentiment)
    X_train_t, X_test_t, y_train_t, y_test_t = Training_Miner().split_train_test_topics(reviews_topics, bow_topics)

    trained_mnb = Training_Miner().train_multinomial_nb(X_train_s, y_train_s)
    trained_cnb = Training_Miner().train_complement_nb(X_train_t, y_train_t)

    return trained_mnb, trained_cnb, reviews_sentiment, reviews_topics

def Predict(new_reviews):
    new_reviews['COMENTÁRIO'] = new_reviews['COMENTÁRIO'].str.replace('!', '.')
    cols = new_reviews.columns.difference(['COMENTÁRIO'])
    comments = new_reviews['COMENTÁRIO'].str.split('.')

    new_reviews =  (new_reviews.loc[new_reviews.index.repeat(comments.str.len()), cols].assign(COMENTÁRIO=list(chain.from_iterable(comments.tolist()))))
    
    nan_value = float("NaN")
    new_reviews.replace("", nan_value, inplace=True)
    new_reviews.dropna(subset = ["COMENTÁRIO"], inplace=True)

    new_reviews_2 = new_reviews.copy()

    trained_mnb, trained_cnb, reviews_sentiment, reviews_topics = RunTraining()
    
    to_predict_preprocessed = Training_Miner().pre_processing(new_reviews)

    to_predict_bow_topics = Training_Miner().bow_new_data(reviews_topics, to_predict_preprocessed)
    to_predict_bow_sentiment = Training_Miner().bow_new_data(reviews_sentiment, to_predict_preprocessed)

    new_prediction_sentiment = Training_Miner().predict(trained_mnb, to_predict_bow_sentiment)
    new_prediction_topics = Training_Miner().predict(trained_cnb, to_predict_bow_topics)

    classificacao = new_prediction_topics.tolist()
    sentimento = new_prediction_sentiment.tolist()

    new_reviews_2 = new_reviews_2.assign(SENTIMENTO=sentimento)
    new_reviews_2 = new_reviews_2.assign(CLASSIFICAÇÃO=classificacao)

    new_reviews_2 = new_reviews_2.rename(columns={'DATA': 'Data', 'COMENTÁRIO': 'Comentário', 'CLASSIFICAÇÃO': 'Classificação', 'SENTIMENTO': 'Sentimento'})

    new_reviews_2['Classificação'] = new_reviews_2['Classificação'].replace([0, 1, 2, 3, 4, 5, 6],
                                                ['Não-Informativo',
                                                   'Funcionalidade',
                                                   'Confiabilidade',
                                                   'Usabilidade',
                                                   'Eficiência',
                                                   'Segurança',
                                                   'Compatibilidade'])
    
    new_reviews_2['Sentimento'] = new_reviews_2['Sentimento'].replace([0, 1],['Ruim', 'Bom'])

    return new_reviews_2

    



