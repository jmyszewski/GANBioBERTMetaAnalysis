# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 15:31:35 2021

@author: Joshua Myszewski, BSE, UW-SMPH '24
"""
import time
import pickle
import collections

from Bio import Entrez
from Bio import Medline
import nltk

from http.client import HTTPException


# You must enter your email and the toolname for Entrez Guideline Compliance

toolname = "PubMedCrawler"

# Topic to be searched
searchterms = "1995:2020[DP] AND Lung Cancer"

# "1995:2020[DP] AND (clinical trial, veterinary[pt] or clinical trial[pt])" Veterinary Med Search


"""
add "and clinical trial[pt] to limit to clinical trials"
add 'and year:year[dp] for year limitations'

"""

# Database Definition
dbase = "pubmed"

# Field to be searched
j = "medicine"
    

def yearpuller(fdata):
    fulldata = []
    data = {}
    for journal in range(0, len(fdata)):
        jdata = fdata[journal]
            
    
            # Addresses the edge case of a journal having a single article
        if len(jdata) == 1:
            jdatarange = [0]
        else:
            jdatarange = range(0, len(jdata))
    
        for record in jdatarange:
            pdata = jdata[record]
            date = str(pdata['DP'])
            year = date[0:4]
            try:
                data[year].append(jdata[record])
            except (KeyError, AttributeError):
                data[year] = list([jdata[record]])
        pdata = fdata[journal]
        pdata = pdata[0]  
                
    return data

def pubmedcrawler(searchterms,dbase,useremail):
    """ Search terms: General search terms for a pubmedsearch
            add "and clinical trial[pt] to limit to clinical trials"
            add 'and year:year[dp] for year limitations'
        Dbase: Which NCBI Database to search
        j: what type of journals to search (i.e. veterinary, anesthesiology, etc.)
        useremail: email which NCBI can contact you at if there is an issue with the crawler
        
        Returns a By journal list of the abstracts found, which also breaks down by year
    """

    
    def flatten(x):
        if isinstance(x, collections.Iterable):
            return [a for i in x for a in flatten(i)]
        else:
            return [x]


    def getjids(j, useremail):
        jtype = j + " AND currentlyindexed[All]"
        Entrez.email = useremail
        Entrez.tool = toolname
        handle = Entrez.esearch(db="nlmcatalog", term=jtype, retmax=2000)
        record = Entrez.read(handle)
        print(" {} journals found for the selected field".format(record["Count"]))
        idlist = record["IdList"]
        JIDlist = list(map(int, idlist))
        return JIDlist


    def getpubmed(terms, dbase, useremail):
        """ Retrieves the Medline data for the specified search parameters """
        
        # determine how many articles there are for the search terms

        Entrez.email = useremail
        Entrez.tool = toolname
        handle = Entrez.egquery(term=terms)
        record = Entrez.read(handle)
        handle.close()
        for row in record["eGQueryResult"]:
            if row["DbName"] == dbase:
                cnt = int(row["Count"])
        time.sleep(0.5)        
    # Fetch the Pubmed IDs
        Entrez.email = useremail
        try:
            handle = Entrez.esearch(db=dbase, term=terms, retmax=cnt)
            record = Entrez.read(handle)
            handle.close()
        except HTTPException as e:
            time.sleep(0.5) 
            print("Network problem: %s" % e)
            print("Second (and final) attempt...")
            handle = Entrez.esearch(db=dbase, term=terms, retmax=cnt)
            record = Entrez.read(handle)
            handle.close()

        idlist = record["IdList"]
        cnt=len(idlist)
        time.sleep(0.5) 
    # get the articles
        if cnt > 200000:
            print("Number of abstracts is very large, requires ID list batching to avoid timeout errror")
            
            for a in range(0,(cnt//200000)+1):
                print('ID List batch ',a+1,'/',(cnt//200000)+1)
                if a == 0:
                    topidlist = idlist
                    del idlist
                if a == (cnt//200000):
                    subidlist = topidlist[a*200000:]
                else:
                    subidlist = topidlist[a*200000:((a+1)*200000)]
                subcnt = len(subidlist)
                if subcnt > 10000: # needs edgecase for when subcnt <= 10000
                    print("number of abstracts in ID list batch is greater than 10,000: Batched Retrieval Necessary")
                    if a == (cnt//200000):
                        subbatchrange = range(0,(subcnt//10000)+1)
                    else:
                        subbatchrange = range(0,(subcnt//10000))
                    for b in subbatchrange:
                        
                        time.sleep(0.5) 
                        print('Retrieving batch ',(b+1),'/',(subcnt//10000))                    
                        handle = Entrez.efetch(db=dbase, id=subidlist, rettype="medline",
                                      retmode="text",retstart = (b * 10000),retmax = 10000 )
                        recordsbatch = Medline.parse(handle)
                        if a == 0 and b == 0:
                            records = list(recordsbatch)
                        else:
                            records.extend(recordsbatch)
                
                            
                            
        elif cnt > 10000:
                    print("number of abstracts is greater than 10,000: Batched Retrieval Necessary")
                    for b in range(0,(cnt//10000)+1):
                        time.sleep(0.5) 
                        print('Retrieving batch ',b,'/',(cnt//10000))
        
                        handle = Entrez.efetch(db=dbase, id=idlist, rettype="medline",
                                       retmode="text",retstart = (b * 10000),retmax = 10000 )
                        recordsbatch = Medline.parse(handle)
                        if b == 0:
                            records = list(recordsbatch)
                        else: 
                            records.extend(recordsbatch)
        else:
            handle = Entrez.efetch(db=dbase, id=idlist, rettype="medline",
                               retmode="text")
            records = Medline.parse(handle)

            records = list(records)

        return records, cnt


# Gather the Journal IDs to be searched

# Define the empty list for the records found

# Searches for the search terms by journal ID

    term = searchterms
    rec, c = getpubmed(term, dbase, useremail)



    fdata = list(filter(None, rec))
    #fdata = yearpuller(fdata)
    return fdata

def abstractretriever(SearchResults):
        FullData = SearchResults
        
        abstractlist = []
        SearchResultstoremove = []
        for textabstract in FullData:
            if 'AB' in textabstract:
                abstractlist.append(textabstract['AB'])
            else:
                SearchResultstoremove.append(textabstract)
        for a in SearchResultstoremove:
            SearchResults.remove(a)
        
        Abstracts=abstractlist
        
         
        return Abstracts, SearchResults
# End of Program
