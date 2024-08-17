#!/usr/bin/env python
# coding: utf-8

# In[14]:


#https://www.youtube.com/watch?v=RQTiyQzowLQ
#https://towardsdatascience.com/conversational-sentiment-analysis-on-audio-data-cd5b9a8e990b
#https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/
#https://towardsdatascience.com/tokenize-text-columns-into-sentences-in-pandas-2c08bc1ca790
#https://medium.com/analytics-vidhya/nlp-with-spacy-tutorial-part-2-tokenization-and-sentence-segmentation-352df790a214
from preprocessors import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import os
get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('run', 'sentiments_per_position.py')
def plot_speech_position(sentiment_blob_speech):
    #average polarity per split
    sentiment_blob_agg2 = sentiment_blob_speech[['type', 'position', 'sentiment_blob']].groupby(['type', 'position'],as_index=False).agg(['mean', 'std', 'count'])
    sentiment_blob_agg2 = sentiment_blob_agg2.droplevel(axis=1, level=0).reset_index()
    
    # Calculate a confidence interval as well.
    sentiment_blob_agg2['ci'] = 1.96 * sentiment_blob_agg2['std'] / np.sqrt(sentiment_blob_agg2['count'])
    sentiment_blob_agg2['ci_lower'] = sentiment_blob_agg2['mean'] - sentiment_blob_agg2['ci']
    sentiment_blob_agg2['ci_upper'] = sentiment_blob_agg2['mean'] + sentiment_blob_agg2['ci']
    sentiment_blob_agg2.head()

    #graph 
    great=sentiment_blob_agg2[sentiment_blob_agg2["type"]=="Great Speech"]
    typical=sentiment_blob_agg2[sentiment_blob_agg2["type"]=="Typical Speech"]

    fig , ax = plt.subplots(1,1,figsize=(10,5))
    x = np.arange(1,11)
    ax.plot(x,great["mean"],label='Great Speech',color='c')
    ax.set_xticks(x)
    ax.fill_between(x, great['ci_lower'], great['ci_upper'], color='c', alpha=.15)

    ax.plot(x,typical["mean"],label='Typical',color='r')
    ax.fill_between(x, typical['ci_lower'], typical['ci_upper'], color='r', alpha=.15)
    ax.legend();
    ax.set_title('Sentiment - TextBlob',fontsize=1);
    ax.set_facecolor("w")
    return


# In[16]:




