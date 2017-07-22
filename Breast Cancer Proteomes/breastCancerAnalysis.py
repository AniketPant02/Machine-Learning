# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 14:24:40 2017
 
@author: Aniket Pant
"""

# Breast Cancer Subclass Classifier - iTRAQ Data Supplicant
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

clinicalData = pd.read_csv('C:\\Users\\Aniket Pant\\Documents\\GitHub\\Machine-Learning\\Breast Cancer Proteomes\\breastcancerproteomes\\clinical_data_breast_cancer.csv')
pam50proteinData = pd.read_csv("C:\\Users\\Aniket Pant\\Documents\\GitHub\\Machine-Learning\\Breast Cancer Proteomes\\breastcancerproteomes\\PAM50_proteins.csv")
cancerProteomeData = pd.read_csv("C:\\Users\\Aniket Pant\\Documents\\GitHub\\Machine-Learning\\Breast Cancer Proteomes\\breastcancerproteomes\\77_cancer_proteomes_CPTAC_itraq.csv")

cancerProteomeData[ AO-A12D.01TCGA'] = cancerProteomeData.AO-A12D.01TCGA.fillna(cancerProteomeData['AO-A12D.01TCGA'].mean())
testData.Fare = testData.Fare.fillna(testData['Fare'].mean())
cancerProteomeData.pivot('RefSeq_accession_number', 'gene_name', 'AO-A12D.01TCGA')

plt.figure(figsize=(9,9))
cancerProteomeDataPivot = cancerProteomeData.pivot('RefSeq_accession_number', 'gene_name', 'AO-A12D.01TCGA')



'''
molecularVariance = pd.DataFrame({'Run Count':runCountGraph, 'MonoOxygenCount':MonoOxygenCountGraph}).set_index('Run Count').to_csv('molecularVariance.csv')
'''
'''
plt.xlabel('Gene Name', size = 15)
plt.ylabel('Tissue Sample', size = 15)
plt.title('Gene Activity', size = 15)
sns.heatmap(cancerProteomeDataPivot, annot=True, fmt='.1f', linewidths=.5, square=True, cmap='Blues_r');
'''