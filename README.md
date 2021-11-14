# GAN-BioBERT Comparison to Meta-Analysis Findings

This is the Repository for all data and code associated with the study "Comparison of Meta-Analysis Findings to GAN-BioBERT Sentiment Analysis of Literature Pertaining to Nerve Blocks in Primary Hip and Knee Arthroplasty: A Brief Technical Report" by Joshua J Myszewski, Emily Klossowski, and Kristopher M Schroeder. 

This code and data is distributed freely under the Apache 2.0 License; All we request is that if you use the GAN-BioBERT algorithm for a published work to please cite the following document: 

    Myszewski, Joshua J., Emily Klossowski, Patrick Meyer, Kristin Bevil, Lisa Klesius, and Kristopher M. Schroeder. "Validating GAN-BioBERT: A Methodology For Assessing Reporting  Trends In Clinical Trials." arXiv preprint arXiv:2106.00665 (2021).

The files present in this repository are as follows:
    
Model_save256B: The Pretrained GAN-BioBERT model for sentiment classification. This can be adapted to classify sentiment for any biomedical text as shown in Example1.py
    
PubMedCrawlerNoField.py: This is a PubMed Data Crawler that uses the NCBI e-utilities to retrieve the medline records for any specified PubMed search; An example of how to use this tool is shown in Example2.py **use of this tool must be in compliance with NCBI usage guidelines and failure to follow these guidelines may results in your IP being blocked access from the NCBI servers.**

Metasearch.py: This is the script used to perform all of the analyses associated with the study "Comparison of Meta-Analysis Findings to GAN-BioBERT Sentiment Analysis of Literature Pertaining to Nerve Blocks in Primary Hip and Knee Arthroplasty: A Brief Technical Report"

RCTTable.csv: This is a table of all the Randomized Controlled Trials used for the 2021 meta-analysis performed by Memtsoudis et. al. (see reference).

ObsTable.csv: This is a table of all the Observational studies used for the 2021 meta-analysis performed by Memtsoudis et. al. (see reference). 

ReferencesList.csv: This is a table of all the references used for the 2021 meta-analysis performed by Memtsoudis et. al. along with PMIDs where available (see reference).




**Reference**

Memtsoudis SG, Cozowicz C, Bekeris J, Bekere D, Liu J, Soffin EM, Mariano ER, Johnson RL, Hargett MJ, Lee BH, Wendel P, Brouillette M, Go G, Kim SJ, Baaklini L, Wetmore D, Hong G, Goto R, Jivanelli B, Argyra E, Barrington MJ, Borgeat A, De Andres J, Elkassabany NM, Gautier PE, Gerner P, Gonzalez Della Valle A, Goytizolo E, Kessler P, Kopp SL, Lavand'Homme P, MacLean CH, Mantilla CB, MacIsaac D, McLawhorn A, Neal JM, Parks M, Parvizi J, Pichler L, Poeran J, Poultsides LA, Sites BD, Stundner O, Sun EC, Viscusi ER, Votta-Velis EG, Wu CL, Ya Deau JT, Sharrock NE. Anaesthetic care of patients undergoing primary hip and knee arthroplasty: consensus recommendations from the International Consensus on Anaesthesia-Related Outcomes after Surgery group (ICAROS) based on a systematic review and meta-analysis. Br J Anaesth. 2019 Sep;123(3):269-287. doi: 10.1016/j.bja.2019.05.042. Epub 2019 Jul 24. PMID: 31351590; PMCID: PMC7678169.

cff-version: 1.2.0
message: "If you use this software, please cite it as below."
authors:
  - family-names: Druskat
    given-names: Stephan
    orcid: https://orcid.org/0000-0003-4925-7248
title: "My Research Software"
version: 2.0.4
doi: 10.5281/zenodo.1234
date-released: 2021-08-11
