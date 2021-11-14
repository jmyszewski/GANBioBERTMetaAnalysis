# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 20:43:05 2021

@author: jmysz
"""

from PubMedCrawlerNoField import pubmedcrawler
from PubMedCrawlerNoField import abstractretriever
import json
from BioBertClassifier import biobertclassifier

import numpy as np
import scipy.stats as st
from collections import Counter
import time
import csv
import pandas as pd 
import string

dbase = 'pubmed' # Database to be searched by the crawler function
useremail = 'jmyszewski@wisc.edu' # Contact info for the user in the event there is a technical difficulty
model_name = 'model_save256B' # Model name for the Sentiment Classifier



tic = time.time()


# Import data from CSVs
# Reference List
references = pd.read_csv('ReferencesList.txt',encoding = 'unicode_escape',sep='\t') 
RCTs = pd.read_csv('RCTTable.csv')
Obs = pd.read_csv('ObsTable.csv')

RCTs.columns = [c.replace(' ', '_') for c in RCTs.columns]
Obs.columns = [c.replace(' ', '_') for c in Obs.columns]


#Gathers and determines sentiment for the Randomized Controlled Trials
abstractlist = []
sentimentlist = []
noabstractrefs = []
for idx,row in RCTs.iterrows():
    
    refnum = row['Reference_#']
    reference = references.iloc[refnum-1]
    print('Analyzing Reference #',idx+1)
    if reference['PMID'] == 0:
       print('PMID Not Found') # Excludes articles not indexed in PubMed
       sentiment = 'n/a'
       sentimentlist.append(sentiment)
       abstractlist.append('n/a')
    else:
        referencesearch = int(reference['PMID']) # PMID for the specific article to be retrieved and analyzed
         
        medlinerecord = pubmedcrawler(referencesearch,dbase,useremail) # Retrieves the medline record for the given PMID (or other search term)
        abstract, results = abstractretriever(medlinerecord) # Retrieves specifically the abstract from the medline record
        if abstract and len(abstract) == 1:
            abstractlist.append(str(abstract))
            sentiment = biobertclassifier(abstract,model_name)
            
            if sentiment == 3: 
                sentiment = 0 # Neutral Sentiment
            elif sentiment == 2:
                sentiment = -1 # Negative Sentiment
            else: 
                sentiment = 1 # Positive Sentiment
                
            sentimentlist.append(sentiment)
        else:
            print(referencesearch)
            referencesearch = input('Enter Pubmed ID of Error Article: ')
            medlinerecord = pubmedcrawler(referencesearch,dbase,useremail)
            abstract, results = abstractretriever(medlinerecord)
            if abstract:
                abstractlist.append(str(abstract))
                sentiment = biobertclassifier(abstract,model_name)
                if sentiment == 3:
                    sentiment = 0
                elif sentiment == 2:
                    sentiment = -1
                else: 
                    sentiment = 1  
                sentimentlist.append(sentiment)
            else:
                print('Abstract still not found') # Excludes articles without an abstract
                noabstractrefs.extend(reference)
                sentiment = 'n/a'
                sentimentlist.append(sentiment)
                abstractlist.append('n/a')
        del abstract, results
    
RCTs['Sentiment'] = sentimentlist
RCTs['Abstract'] = abstractlist
AllRCTs = RCTs
RCTs = RCTs[RCTs.Sentiment != 'n/a']

del abstractlist, sentimentlist

#Gathers and determines sentiment for the Randomized Controlled Trials

abstractlist = []
sentimentlist = []

for idx,row in Obs.iterrows():

    refnum = row['Reference_#']
    reference = references.iloc[refnum-1]
    print('Analyzing Reference #',idx+1)
    if reference['PMID'] == 0:
       print('PMID Not Found')
       sentiment = 'n/a'
       sentimentlist.append(sentiment)
       abstractlist.append('n/a')
    else:
        referencesearch = int(reference['PMID'])
        medlinerecord = pubmedcrawler(referencesearch,dbase,useremail)
        abstract, results = abstractretriever(medlinerecord)
        if abstract and len(abstract) == 1:
            abstractlist.append(str(abstract))
            sentiment = biobertclassifier(abstract,model_name)
            if sentiment == 3:
                sentiment = 0
            elif sentiment == 2:
                sentiment = -1
            else: 
                sentiment = 1
    
            sentimentlist.append(sentiment)
        else:
            print(referencesearch)
            referencesearch = input('Enter Pubmed ID of Error Article: ')
            medlinerecord = pubmedcrawler(referencesearch,dbase,useremail)
            abstract, results = abstractretriever(medlinerecord)
            if abstract:
                abstractlist.append(str(abstract))
                sentiment = biobertclassifier(abstract,model_name)
                if sentiment == 3:
                    sentiment = 0
                elif sentiment == 2:
                    sentiment = -1
                else: 
                    sentiment = 1 
                sentimentlist.append(sentiment)
            else:
                noabstractrefs.extend(reference)
                print('Abstract still not found')
                noabstractrefs.extend(reference)
                sentiment = 'n/a'
                sentimentlist.append(sentiment)
                abstractlist.append('n/a')
        del abstract, results
    
Obs['Sentiment'] = sentimentlist
Obs['Abstract'] = abstractlist

AllObs = Obs
Obs = Obs[Obs.Sentiment != 'n/a']

abstractlist = []
sentimentlist = []
noabstractrefs = []

## Build Subgroups for analysis 

lowbiasObs = Obs[(Obs.A != 'High') & (Obs.B != 'High') & (Obs.C != 'High') & (Obs.D != 'High')]
highbiasObs = Obs[(Obs.A == 'High') | (Obs.B == 'High') | (Obs.C == 'High') | (Obs.D == 'High')]

lowbiasRCTs = RCTs[(RCTs.A != 'High') & (RCTs.B != 'High') & (RCTs.C != 'High') & (RCTs.D != 'High') & (RCTs.E != 'High') & (RCTs.F != 'High') & (RCTs.G != 'High')]
highbiasRCTs = RCTs[(RCTs.A == 'High') | (RCTs.B == 'High') | (RCTs.C == 'High') | (RCTs.D == 'High') | (RCTs.E == 'High') | (RCTs.F == 'High') | (RCTs.G == 'High')]

NAandGeneralOBs = Obs[Obs.Study_Technique.str.contains('NA') & Obs.Study_Technique.str.contains('GA') ]
NAandGeneralRCTs = RCTs[RCTs.Study_Technique.str.contains('NA') & RCTs.Study_Technique.str.contains('GA') ]

NAonlyOBs = Obs[Obs.Study_Technique.str.contains('NA') & ~Obs.Study_Technique.str.contains('GA') ]
NAonlyRCTs = RCTs[RCTs.Study_Technique.str.contains('NA') & ~RCTs.Study_Technique.str.contains('GA') ]

GAonlyOBs = Obs[~Obs.Study_Technique.str.contains('NA') & Obs.Study_Technique.str.contains('GA') ]
GAonlyRCTs = RCTs[~RCTs.Study_Technique.str.contains('NA') & RCTs.Study_Technique.str.contains('GA') ]

KneeObs = Obs[Obs.Study_Technique.str.contains('TKA')]
HipObs = Obs[Obs.Study_Technique.str.contains('THA')]

KneeRCTs = RCTs[RCTs.Study_Technique.str.contains('TKA')]
HipRCTs = RCTs[RCTs.Study_Technique.str.contains('THA')]

patientnumbers = list(Obs['Patients'])
patientnumbers.extend(list(RCTs['Patients']))

medianptcount = np.median(patientnumbers)

highptRCTs = RCTs[(RCTs.Patients > medianptcount)]
lowptRCTs = RCTs[(RCTs.Patients <= medianptcount)]
highptObs = Obs[(Obs.Patients > medianptcount)]
lowptObs = Obs[(Obs.Patients <= medianptcount)]

RetroObs = Obs[(Obs.Study_Type.str.contains('Retrospective'))]
ProspObs = Obs[(Obs.Study_Type.str.contains('Prospective'))]
CCObs = Obs[(Obs.Study_Type.str.contains('Case-Control'))]

SubgroupResults = {}

def resultcompiler(subgroup,subgroup2=pd.DataFrame()):
    subgroupsentiment = list(subgroup['Sentiment'])
    if not subgroup2.empty:
        subgroupsentiment.extend(list(subgroup2['Sentiment']))
    samplesize = len(subgroupsentiment)
    meansent = np.mean(subgroupsentiment)
    ci = st.t.interval(alpha=0.95, df=len(subgroupsentiment)-1,loc=np.mean(subgroupsentiment),scale=st.sem(subgroupsentiment))
    results = (samplesize,meansent,ci,st.sem(subgroupsentiment))
    return results

# by study type
# Sample Size, Mean, CI, SEM 
SubgroupResults['RCTs']=resultcompiler(RCTs)
SubgroupResults['Obs']=resultcompiler(Obs)
SubgroupResults['All Studies'] = resultcompiler(RCTs,Obs)
SubgroupResults['RetroObs']=resultcompiler(RetroObs)
SubgroupResults['ProspObs']=resultcompiler(ProspObs)

# by sample size
SubgroupResults['HighPtCount']=resultcompiler(highptObs,highptRCTs)
SubgroupResults['LowPtCount']=resultcompiler(lowptObs,lowptRCTs)

# by bias risk
SubgroupResults['HighBias']=resultcompiler(highbiasObs,highbiasRCTs)
SubgroupResults['LowBias']=resultcompiler(lowbiasObs,lowbiasRCTs)

# By procedure type
SubgroupResults['Knee']=resultcompiler(KneeObs,KneeRCTs)
SubgroupResults['Hip']=resultcompiler(HipObs,HipRCTs)

# By Anesthesia type
SubgroupResults['GAOnly']=resultcompiler(GAonlyOBs,GAonlyRCTs)
SubgroupResults['NAOnly']=resultcompiler(NAonlyOBs,NAonlyRCTs)
SubgroupResults['GA+NA']=resultcompiler(NAandGeneralOBs,NAandGeneralRCTs)

toc = time.time()
print(toc-tic, 'sec Elapsed during subgroup analysis')




