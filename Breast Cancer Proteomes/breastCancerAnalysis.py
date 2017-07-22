# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 14:24:40 2017
 
@author: Aniket Pant
"""

# Breast Cancer Subclass Classifier - iTRAQ Data Supplicant
import pandas as pd
import matplotlib.pyplot as plt

clinicalData = pd.read_csv('C:\\Users\\Aniket Pant\\Documents\\GitHub\\Machine-Learning\\Breast Cancer Proteomes\\breastcancerproteomes\\clinical_data_breast_cancer.csv')
pam50proteinData = pd.read_csv("C:\\Users\\Aniket Pant\\Documents\\GitHub\\Machine-Learning\\Breast Cancer Proteomes\\breastcancerproteomes\\PAM50_proteins.csv")
cancerProteomeData = pd.read_csv("C:\\Users\\Aniket Pant\\Documents\\GitHub\\Machine-Learning\\Breast Cancer Proteomes\\breastcancerproteomes\\77_cancer_proteomes_CPTAC_itraq.csv")


