#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os
import shap
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm
from sklearn import metrics
from preprocessors import *
from sklearn.metrics import *
from datetime import datetime
import matplotlib.pyplot as plt
from Dataset import SpeechDataset
from DataLoader import DataLoader
from xgboost import XGBClassifier
from sklearn.preprocessing import *
from matplotlib.pyplot import figure
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

cwd = os.getcwd()
#usr_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36'
#base_url='https://www.americanrhetoric.com/'
speeches_dir = f"{cwd}/speeches/"
#scrape_speeches(base_url, speeches_dir, usr_agent)


#dataset_types = ["important", "typical"]
dataset_types = ["important"]
resources_dir = f"{cwd}/resources"
saving_dir = f"{cwd}/resources/dataset_all.csv"

from tqdm import tqdm

if not os.path.exists(saving_dir):
    df_dict = {
        "speaker": [], "title": [], "year": [], "content": [],"fear": [],
        "anger": [], "disgust": [], "disgust": [], "joy": [], "neutral": [], "sadness": [], "surprise": [],
        "polarity": [], "subjectivity": [], "complexity": [], "lexical_richness": [],
        "entities_proportion_in_speech": [], "imagery_proportion_in_speech": [],
        "stopwords_proportion_in_speech": [], "mean_sentence_length": [],
        "label": []
    }

    imagery_words = pd.read_csv("resources/visual_words.csv", header=None)
    imagery_words = list(imagery_words[0].array)
    stop_words = list(spacy.load("en_core_web_md").Defaults.stop_words)

    for dataset_type in dataset_types:
        path = f"{cwd}/dataset/{dataset_type}"
        dataset = SpeechDataset(path)
        dataloader = DataLoader(dataset)
        with tqdm(total=len(dataloader.dataset)) as progress_bar:
            for speech in dataloader:
                for key in df_dict.keys():
                    try:
                        df_dict[key].append(getattr(speech, f"get_{key}")())
                    except:
                        pass
                emotions = speech.get_emotion_scores(return_all_scores=True)[0]
                for emotion in emotions:
                    df_dict[emotion["label"]].append(emotion["score"])

                df_dict["entities_proportion_in_speech"].append(speech.get_proportion_in_speech(speech.get_entities()))
                df_dict["imagery_proportion_in_speech"].append(speech.get_proportion_in_speech(imagery_words))
                df_dict["stopwords_proportion_in_speech"].append(speech.get_proportion_in_speech(stop_words))
                if dataset_type == "important":
                    df_dict["label"].append(1.0)
                else:
                    df_dict["label"].append(0.0)
                progress_bar.update(1)

    if not os.path.exists(resources_dir):
        os.mkdir(resources_dir)
    df = pd.DataFrame(df_dict)
    df.to_csv(saving_dir)
else:
    df = pd.read_csv(saving_dir)

# to trasnform the sppech dictonary into a df.
df = pd.DataFrame.from_dict(df_dict, orient='index')
df = df.transpose()
df.to_csv(saving_dir)

