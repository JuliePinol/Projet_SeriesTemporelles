#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:19:29 2020

@author: juliepinol
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime 
import pywt
from pandas.plotting import register_matplotlib_converters
from sklearn.preprocessing import normalize
register_matplotlib_converters()



# -------------  Generation de la transformation en ondelettes -------------------

# si on veut savoir les ondelettes disponibles : 
#print([name for family in pywt.families() for name in pywt.wavelist(family)])

def ondelettes(liste_jours):
    #Creation de la matrice des coefficients de la transformation
    matrice_coeffs = []
    matrice_bruits_cD1 = []
    matrice_bruits_cD2 = []
    matrice_bruits_cD3 = []

    #index_jour= []
    #traitement des jours au cas par cas:
    for jour in liste_jours:
       # print(jour['day'][0])
        #index_jour.append(jour['day'][0])'
        coeffs = pywt.wavedec(jour['temperature'], 'db5', 'reflect', level=2)
        matrice_coeffs.append(coeffs[0]) #on ajoute à la matrice la composante basale du signal
        #for i in range(1,len(coeffs)): 
            #matrice_bruit.append(coeffs[i])
        matrice_bruits_cD3.append(coeffs[1])
        matrice_bruits_cD2.append(coeffs[2])
        #matrice_bruits_cD1.append(coeffs[3])
    #matrice_coeffs_df = pd.DataFrame(matrice_coeffs)
    #print(index_jour)
    matrice_coeffs_norm = normalize(matrice_coeffs)
    return matrice_coeffs_norm

'''
# -------------  Calcul des distances -------------------
### FONCTION NON UTILISEE
    
def calcul_distances (matrice):
    
    matrice_distance = []
    
    #chaque ligne correspond aux coefficients S d'un jour
    for indice_ligne in range(len(matrice)) :
        #pour toutes les lignes inférieures on calcule la distance
        for indice_lignes_restantes in range (indice_ligne, len(matrice)):
            distance = 0
            ligne_distance = [0]*indice_ligne
            #pour chaque vecteur coefficent on calcule la norme matricielle
            #print(matrice[indice_ligne])
            for nombre in range(len(matrice[indice_ligne])):
                distance = distance + (matrice[indice_ligne][nombre]-matrice[indice_lignes_restantes][nombre])**2
            distance = distance**(1/2)
            ligne_distance.append(distance)
        #on met à jour la matrice des distances (matrice triangulaire supérieure)
        matrice_distance.append(ligne_distance)
    #print(matrice_distance)
        
    return matrice_distance 

'''
# -------------  Clustering Hierarchique -------------------


#librairie pour la CAH
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
     
def clustering_CAH(matrice)  :
    
    #calcul de la matrice des distances réduites à mettre en entrée de la méthode CAH
    matrice_d = pdist(matrice, metric= 'seuclidean')
    
    #Creation du dataframe
    matrice_df = pd.DataFrame(matrice)
    
    #construction de la matrice de lien ie la typologie du CAH
    Z = linkage(matrice_d,method='ward',metric='seuclidean')

    #affichage du dendrogramme
    plt.title("CAH")
    dendrogram(Z,orientation='top',color_threshold=0,leaf_rotation=90)
    plt.show()

    #et matérialisation des 4 classes (hauteur de coupure height = 40)
    plt.title('CAH avec matérialisation des 4 classes')
    dendrogram(Z,orientation='top',color_threshold=55,leaf_rotation=90) 
    plt.show()

    #découpage à la hauteur = 40 
    groupes_cah = fcluster(Z,t=55,criterion='distance')
    
    #index triés des groupes
    #idg = np.argsort(groupes_cah)

    #affichage des observations et leurs groupes
    #groupes = pd.DataFrame(matrice_df.index[idg],groupes_cah[idg])
    groupes_cah1 = pd.DataFrame(groupes_cah)
    #index_j = pd.Dataframe(index)
    #result = pd.merge(index_j, groupes_cah1)
    #print(result)
    groupes_cah1.to_csv('/Users/juliepinol/Desktop/Test_cluster_saumon_epures_norm.csv',index = None, header=False)             

'''
def clustering_CAH_bruits(matrice):
    #calcul de la matrice des distances réduites à mettre en entrée de la méthode CAH
    matrice_d = pdist(matrice, metric= 'seuclidean')
    
    #Creation du dataframe
    matrice_df = pd.DataFrame(matrice)
    
    #construction de la matrice de lien ie la typologie du CAH
    Z = linkage(matrice_d,method='ward',metric='seuclidean')

    #affichage du dendrogramme
    plt.title("CAH")
    dendrogram(Z,orientation='top',color_threshold=0,leaf_rotation=90)
    plt.show()

    #et matérialisation des 4 classes (hauteur de coupure height = 40)
    plt.title('CAH avec matérialisation des 7 classes')
    dendrogram(Z,orientation='top',color_threshold=40,leaf_rotation=90) 
    plt.show()

    #découpage à la hauteur = 40 
    groupes_cah = fcluster(Z,t=40,criterion='distance')
    
    #index triés des groupes
    #idg = np.argsort(groupes_cah)

    #affichage des observations et leurs groupes
    #groupes = pd.DataFrame(matrice_df.index[idg],groupes_cah[idg])
    groupes_cah1 = pd.DataFrame(groupes_cah)
    #index_j = pd.Dataframe(index)
    #result = pd.merge(index_j, groupes_cah1)
    #print(result)
    groupes_cah1.to_csv('/Users/juliepinol/Desktop/Test_cluster_saumon_epures.csv',index = None, header=False)             
 '''
   
# -------------  Importation des données et Code Principal -------------------


#raw_data = pd.read_csv('C:/Users/kevin/OneDrive/0Main/Cours APT/3A - IODAA/Projet fil rouge/Scripts python/saumon_frigo_MPday1.csv', sep = ';')

#raw_data = pd.read_csv('/Users/juliepinol/Desktop/saumon_frigo_MP_jours_complets_epure.csv', sep = ';')


raw_data_1 = pd.read_csv('/Users/juliepinol/Desktop/saumon_frigo_MP_jours_complets.csv', sep = ';')
raw_data_1['temperature'] = raw_data_1['temperature'].str.replace(',','.').apply(float)
raw_data_1['temperature']=normalize(raw_data_1['temperature'])
# Create x, where x the 'scores' column's values as floats
x = df[['score']].values.astype(float)
# Create a minimum and maximum processor object
min_max_scaler = preprocessing.MinMaxScaler()

# Create an object to transform the data to fit minmax processor
x_scaled = min_max_scaler.fit_transform(x)

# Run the normalizer on the dataframe
df_normalized = pd.DataFrame(x_scaled)

raw_data_2 = pd.read_csv('/Users/juliepinol/Desktop/fruits_ch_positive_complets.csv', sep = ';')
raw_data_2['temperature'] = raw_data_2['temperature'].str.replace(',','.').apply(float)
raw_data_2['temperature']=normalize(raw_data_2['temperature'])


raw_data_3 = pd.read_csv('/Users/juliepinol/Desktop/patisserie_jours_complets.csv', sep = ';')
raw_data_3['temperature'] = raw_data_3['temperature'].str.replace(',','.').apply(float)
raw_data_3['temperature']=normalize(raw_data_3['temperature'])

raw_data_4 = pd.read_csv('/Users/juliepinol/Desktop/soja_positif_jours_complets.csv', sep = ';')
raw_data_4['temperature'] = raw_data_4['temperature'].str.replace(',','.').apply(float)
raw_data_4['temperature']=normalize(raw_data_4['temperature'])


raw_data_5 = pd.read_csv('/Users/juliepinol/Desktop/saumon_jours_complets.csv', sep = ';')
raw_data_5['temperature'] = raw_data_5['temperature'].str.replace(',','.').apply(float)
raw_data_5['temperature']=normalize(raw_data_5['temperature'])



raw_data = pd.concat([raw_data_1, raw_data_2,raw_data_3,raw_data_4,raw_data_5], ignore_index=True)
print(raw_data)
#conversion des jours au format date
raw_data['day'] = pd.to_datetime(raw_data['day'])

#remplacement des virgules par des points
raw_data['temperature'] = raw_data['temperature'].str.replace(',','.').apply(float)

#création d'une liste de dataframes contenant chacun les données d'un jour
liste_jours = [group[1] for group in raw_data.groupby(raw_data.set_index('day').index.date)]
#print(raw_data.set_index('day').index.date)
#matrice_test = ondelettes(liste_jours)
#mat_distance = calcul_distances(matrice_test)
#clustering_CAH(matrice_test)


