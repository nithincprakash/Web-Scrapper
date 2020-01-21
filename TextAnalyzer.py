# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 15:44:09 2018

@author: mankayarkarasi.c
"""


import os
import sys
import inflect
import pyodbc
import pandas as pd
from sklearn.pipeline import Pipeline
from nltk import pos_tag,word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from watson_developer_cloud import NaturalLanguageUnderstandingV1
#import watson_developer_cloud.natural_language_understanding.features.v1 as Features
from watson_developer_cloud.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions, SentimentOptions, EmotionOptions, ConceptsOptions
from googletrans import Translator


#Lemmatizer for converting words to root words
lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize('process')
#Inflect engine to convert plurals to singular
stemmerEngine = inflect.engine()


def senti_rating(IBM_senti_score):
    if(IBM_senti_score > 0.60):
        return [5,'Positive']
    elif(IBM_senti_score > 0.10 and IBM_senti_score < 0.60):
        return [4,'Positive']
    elif(IBM_senti_score > -0.10 and IBM_senti_score < 0.10):
        return [3,'Neutral']
    elif(IBM_senti_score > -0.60 and IBM_senti_score < -0.10):
        return [2,'Negative']
    elif(IBM_senti_score < -0.60):
        return [1,'Negative']

def sentiment_only(IBM_senti_score):
    if(IBM_senti_score > 0.60):
        return 'Very Satisfied'
    elif(IBM_senti_score > 0.10 and IBM_senti_score < 0.60):
        return 'Satisfied'
    elif(IBM_senti_score > -0.10 and IBM_senti_score < 0.10):
        return 'Neutral'
    elif(IBM_senti_score > -0.60 and IBM_senti_score < -0.10):
        return 'Dissatisfied'
    elif(IBM_senti_score < -0.60):
        return 'Very Dissatisfied'

vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)


ospath = os.getcwd()
sys.path.append(ospath)

## Function for translation ===================================================

translator = Translator()
def translator_method(text, lang, translator):
    translation = translator.translate(text, dest=lang)
    review = translation.text
    return review

#==============================================================================


#def connect_db_CXCalibration():
#    connection = pyodbc.connect(
#                    'DRIVER={ODBC Driver 13 for SQL Server};'
#                    'SERVER=cxmvpdb.database.windows.net;'
#                    'DATABASE=CXCalibration;'
#                    'UID=cxadmin;'
#                    'PWD=Evry@1234;'    
#                    )
#    return connection
#
#
#connection = connect_db_CXCalibration()
#cursor = connection.cursor()
#cursor.execute("SELECT * FROM cx_keys WHERE key_name = 'ibm_watson'")
#keys = cursor.fetchone()
#username = keys[2]
#password = keys[3]
#connection.close()


class textAnalyzer:
    
#    username = username
#    password = password
    
    def __init__(self, review):
        self.review = review
        nlu = NaturalLanguageUnderstandingV1(
                                        version='2018-03-16',
                                        iam_apikey='NjcHefiexvwtga7eVrEm43WUXrVvf_ABOHoQxnSt5mBC',
                                        url = 'https://gateway-lon.watsonplatform.net/natural-language-understanding/api'
                                        )
        #self.response = natural_language_understanding.analyze(text=review.lower(),language='en', features=[Features.Sentiment(), Features.Keywords(), Features.Emotion(),  Features.Entities(), Features.SemanticRoles()]) 
        #self.response = natural_language_understanding.analyze(text=self.review.strip(), language='en', features= Features(entities=EntitiesOptions(), keywords=KeywordsOptions(), sentiment = SentimentOptions(), emotion=EmotionOptions()))
        self.response = nlu.analyze(text = review.strip(), language='en',
            features = Features(
                        entities=EntitiesOptions(),
                        keywords=KeywordsOptions(),
                        sentiment=SentimentOptions(),
                        emotion=EmotionOptions(),
                        concepts=ConceptsOptions() 
                        )).get_result()
        
    #Method to return noun keywords
    def keywords(self):
        print(self.response)
        keywords = self.response['keywords']
        #print(keywords)
        self.keyword_list = []
        tags = []
        keyTags = []
        
        for keyword_dict in keywords:
            if(keyword_dict['relevance'] > 0.50):
                self.keyword_list.append(keyword_dict['text'])
                # Parts of Speech Tagging
                tags.append(pos_tag(word_tokenize(keyword_dict['text'])))
        
        for i in range(0, len(tags)):
                for tuples in tags[i]:
                    if tuples[1] == "NN" or tuples[1] == "NNS":
                        try:
                            keyTags.append(lemmatizer.lemmatize(tuples[0]))
                        except Exception as ex:
                            print("tuple: ",tuples)
                            print(ex)
        
        #Removing duplicates within the set of keywords
        keyTags = list(set(keyTags))        
        
        #Converting Words to singular form
        final_keywords = []
        for keyTag in keyTags:
            for i in range(0, len(self.keyword_list)):
                if (keyTag in self.keyword_list[i]):
                    singular_word = stemmerEngine.singular_noun(self.keyword_list[i])
                    if(singular_word):
                        final_keywords.append(singular_word)
                    else:
                        final_keywords.append(self.keyword_list[i])
        
        final_keywords = list(set(final_keywords))
        return final_keywords
        
    def adjectives(self):
            keyWord_list = self.keyword_list        
            adjDict = {}
            for i in range(0, len(keyWord_list)):
                pos_tags = pos_tag(word_tokenize(keyWord_list[i]))
                singular_word = stemmerEngine.singular_noun(keyWord_list[i])
                if (singular_word):
                    singular = singular_word
                else:
                    singular = keyWord_list[i]
                for tuples in pos_tags:
                   #JJ, JJS, JJR are adjevtive tags 
                   if tuples[1] == "JJ" or tuples[1] == "JJS" or tuples[1] == "JJR":
                       adjDict.setdefault(singular, []).append(lemmatizer.lemmatize(tuples[0]))                
            return adjDict
    
    def emotions(self):
        if 'emotion' in self.response.keys():
            emotion = self.response['emotion']['document']['emotion']
        else:
            emotion = []
        return emotion
    def concepts(self):
        if 'concept' in self.response.keys():
            concept = self.response['emotion']['document']['emotion']
        else:
            concept = []
        return concept
    def sentiments(self):
        sentiments = self.response['sentiment']
        sentiment_rating = senti_rating(sentiments['document']['score'])
        return sentiment_rating
    def sentiments_new(self):
        sentiments = self.response['sentiment']
        sentiment_rating = sentiment_only(sentiments['document']['score'])
        return sentiment_rating
    
    


#def topic_detection(keyword_list):
#   prediction = list(vec_forest.predict(keyword_list))
#   prediction_scores = list(vec_forest.predict_proba(keyword_list))
#   final_prediction =[]
#   for i in range(0, len(prediction)):
#       score_list = list(prediction_scores[i])
#       if (max(score_list) > 0.59):
#           final_prediction.append(prediction[i])
#       else:
#           final_prediction.append('Others')
#   return final_prediction