#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import spacy
import math
from textblob import TextBlob
#from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np

def blob_sentiment(txt):
    sent = TextBlob(txt).sentiment.polarity
    return sent

cwd = os.getcwd()
sentiment_blob_speech = f"{cwd}/resources/sentiment_blob_speech.csv"
if not os.path.exists(sentiment_blob_speech):
    dataset_all = f"{cwd}/resources/dataset_all.csv"
    df= pd.read_csv(dataset_all)
    df['label'] = df['label'].astype('str') 
    df["label"].str.strip()
    df["label"].replace({"1.0": "Great Speech", "0.0": "Typical Speech"}, inplace=True)
    df = df.rename(columns = {'Unnamed: 0':'speech_i'})
    nlp = spacy.load("en_core_web_md")
    stop_words = nlp.Defaults.stop_words
        
    # use a for loop for all speeches. add the # and type.
    sentiment_blob_speech = pd.DataFrame( columns=['speech_number', 'type','position', 'sentiment_blob'])
    df_speech_10sp=[]

    for index, row in df.iterrows():
        df_speech = []
        df_speech_10sp=[]
        cuts=0
        txt=' '.join(row["content"].split('\n'))
        sentences=[sent.text for sent in nlp(txt).sents]

        #remove stop words
        for sent in sentences:
            doc=nlp(sent)
            word_tokens = [token.text for token in doc]  
            filtered_sentence = [w for w in word_tokens if not w in stop_words]  
            x = ''
            for word in filtered_sentence:
                x +=' '+word
            df_speech.append(x)

        #mix sentences into 10 splits
        cuts=math.floor(len(df_speech)/10)

        for i in range(9):
            df_speech_10sp.append(''.join(df_speech[i*cuts:(i+1)*cuts]))
        df_speech_10sp.append(''.join(df_speech[(i+1)*cuts+1:]))

        df_sentiment = pd.DataFrame(df_speech_10sp)
        df_sentiment.columns = ['text']
        df_sentiment['sentiment_blob'] = ''
        df_sentiment.reset_index(inplace=True)
        df_sentiment = df_sentiment.rename(columns = {'index':'position'})
        df_sentiment['sentiment_blob'] = df_sentiment['text'].apply(lambda x : blob_sentiment(x))
        df_sentiment['speech_number']=row["speech_i"]
        df_sentiment['type']=row["label"]
    #    sentiment_blob_speech=sentiment_blob_speech.append(df_sentiment[['speech_number', 'type','position', 'sentiment_blob']])
        sentiment_blob_speech = pd.concat([sentiment_blob_speech, df_sentiment[['speech_number', 'type','position', 'sentiment_blob']]])
    
    df.to_csv(sentiment_blob_speech)
    print("sentiment_blob_speech created")
    

