# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 18:12:01 2021

@author: jmysz
"""

from PubMedCrawlerNoField import pubmedcrawler
from PubMedCrawlerNoField import abstractretriever

dbase = 'pubmed' # Database to be searched by the crawler function
useremail = '' # email for the user in the event there is a technical difficulty this is required for NCBI Entrez Guideline compliance


search = "comparison of neuraxial anesthesia to general anesthesia in total knee arthroplasty and clinical trial[pt]"
"""
add "and clinical trial[pt] to limit to clinical trials"
add 'and year:year[dp] for year limitations'

"""

medlineresults = pubmedcrawler(search,dbase,useremail) # retrieves the medline record for the given 

abstract, results = abstractretriever(medlineresults) # retrieves just the abstracts from the medline records and returns them as a list
