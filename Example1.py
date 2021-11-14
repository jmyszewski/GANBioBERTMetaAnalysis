# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 18:02:29 2021

@author: jmysz
"""

from BioBertClassifier import biobertclassifier

model_name = 'model_save256B' # Model name for the Sentiment Classifier

abstract = ["This study found that the treatment in question was not indicated for mechanical koilonychiectomy surgery"]# Example text to classify the sentiment of
sentiment = biobertclassifier(abstract,model_name)
            
if sentiment == 3: 
    sentiment = 0 # Neutral Sentiment            
elif sentiment == 2:
    sentiment = -1 # Negative Sentiment
else:
    sentiment = 1 # Positive Sentiment

''' the raw output of the sentimentclassifier (biobertclassifier) is a value of 
3 for neutral, 
2 for negative,
1 for positive as a result of the algorithms structure but this is easily adapted to a more intuitive number '''
    
# This algorithm is also able to process a number of abstracts at once


abstract.extend(abstract)
abstract.extend(abstract)

sentimentlist = biobertclassifier(abstract,model_name) 

for a in range(0,len(sentimentlist)):
        if sentimentlist[a] == 3:
            sentimentlist[a] = 0
        elif sentimentlist[a] == 2:
            sentimentlist[a] = -1
        else: 
            sentimentlist[a] = 1
