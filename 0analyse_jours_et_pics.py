# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 15:09:03 2020

@author: kevin
"""

import pandas as pd
from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt
import numpy as np
import datetime 
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#paramètres des datasets
time_interval = 10 #temps en minutes -- valeur utilisée pour la conversion de la largeur des pics

# ------------- Recherche des pics: find_peaks -------------------
#on utilise la fonction find_peaks de scipy.signal:

#liste des paramètres disponibles (liste non exhaustive)
#threshold = différence verticale minimale entre deux pics successifs
#threshold = 0.00
#hauteur des pics = distance entre la valeur zéro et le point le plus élevé du pic (=SEUIL)
#idéalement ce serait la température de consigne ou une valeur un peu plus haute
### PARAMETRE A REGLER
#hauteur = 1 #il est reglé plus bas dans le code désormais
#distance = distance horizontale minimale entre deux pics
#prominence = "distance minimale à descendre depuis un pic pour arriver de nouveau à une remontée"
#permet de ne garder que les pics significatifs dans un groupement de pics ajdacent (i.e. d'éliminer le bruit)
#est-ce un critère utile pour notre cas de figure?

## => Cette fonction est utilisée dans la boucle


#raw_data = pd.read_csv('C:/Users/kevin/OneDrive/0Main/Cours APT/3A - IODAA/Projet fil rouge/Scripts python/saumon_frigo_MPday1.csv', sep = ';')
raw_data = pd.read_csv('/Users/juliepinol/Desktop/fruits_ch_positive_complets.csv', sep = ';')

#conversion des jours au format date
raw_data['day'] = pd.to_datetime(raw_data['day'])

#remplacement des virgules par des points
raw_data['temperature'] = raw_data['temperature'].str.replace(',','.').apply(float)

#création d'une liste de dataframes contenant chacun les données d'un jour
liste_jours = [group[1] for group in raw_data.groupby(raw_data.set_index('day').index.date)]

#liste des informations des pics qui vont être stockés
list_peak_time = [] 
list_dbt_pic = []
list_fin_pic=[]
list_peak_height=[]
list_peak_width=[]
list_peak_area_diff=[]
list_pente=[]

#liste info sur les jours
list_dates=[]
list_nb_pics=[]
list_t_moy=[]
list_t_max=[]
list_t_min=[]
list_ecart_t=[]

#paramètres de recherche des pics
hauteur = raw_data['temperature'].median() + 0.5

#traitement des jours au cas par cas:
for jour in liste_jours:
	
	
	#obtention des pics
	x = jour['temperature']
	peaks, properties = find_peaks(x, height = hauteur)
	#l'index du 1er instant de chaque jour varie selon la position dans le dataset 1er jour: index= 0, deuxieme jour = 144, etc
	premier_index = jour.index[0]
	peak_indexes = [peak_index + premier_index for peak_index in peaks]
	
	#analyse du jour
	date_du_jour = jour['temps'][premier_index][0:10]
	nb_pics = len(peaks)
	t_moy = jour['temperature'].mean()
	t_max = jour['temperature'].max()
	t_min = jour['temperature'].min()
	ecart_t = t_max - t_min
	
	list_dates+=[date_du_jour]
	list_nb_pics+=[nb_pics]
	list_t_moy+=[t_moy]
	list_t_max+=[t_max]
	list_t_min+=[t_min]
	list_ecart_t+=[ecart_t]
	
	# ------------- calcul de la largeur des pics: peaks_widths (du module scipy.signal) -------------------
	
	widths, with_heights, left_ips, right_ips = peak_widths(x, peaks)
	#left_ips[k] et right_ips[k] sont les coordonnées de début et fin de pic du k-ème pic
	#print('largeur des signaux', widths)
	
	
	# ------------ affichage graphique ------------
#	graph = plt.plot(x)
#	plt.plot(peak_indexes, x[peak_indexes], "x")
#	plt.plot(np.zeros_like(x), "--", color="gray")
#	plt.show()
	
	# stockage des informations sur les peaks
	for k in range(len(peak_indexes)):
		peak_index = peak_indexes[k] #index du pic dans la série temporelle initiale
		# récupération de la date et heure du pic dans le jour 
		peak_time = jour['temps'][peak_index]
		peak_height = properties['peak_heights'][k] - hauteur #différence entre le seuil de détection du pic et la hauteur du pic
		peak_width = widths[k] * time_interval
		
		#estimation de l'aire entre la courbe de température et la hauteur minimale de détection des pics
		#cela permet de voir l'amplitude du dépassement de température par rapport à ce qui est "acceptable"
		dbt_pic = int(left_ips[k]) 
		fin_pic = int(right_ips[k])
		index_pic = [k*time_interval for k in range(dbt_pic,fin_pic+1)]
		area_under_minimal_height = hauteur*(fin_pic-dbt_pic)*time_interval
		temperatures = jour['temperature'].tolist()
		area_under_peak = np.trapz(temperatures[dbt_pic:fin_pic+1],index_pic) #integration selon la "composite trapezoidale rule" voir documento de la fonction en ligne
		area_diff = area_under_peak - area_under_minimal_height
		
		#calcul de la pente entre le début du pic et le pic
		index_pic = peaks[k]
		valeur_dbt_pic = temperatures[dbt_pic]
		valeur_fin_pic = temperatures[index_pic]
		if index_pic-dbt_pic !=0:
			pente = (valeur_fin_pic - valeur_dbt_pic)/((index_pic - dbt_pic)*10)
		else:
			pente = 'NA'
		
		list_peak_time+= [peak_time] 
		list_peak_height+= [peak_height]
		list_peak_width+= [peak_width]
		list_peak_area_diff+= [area_diff]
		list_pente+= [pente]
		list_dbt_pic+=[(jour['temps'][premier_index+dbt_pic])]
		list_fin_pic+=[(jour['temps'][premier_index+fin_pic])]

 
#creation des dataframe qui vont être transformés en CSV
#dataframe des jours
dico_jour = {'date':list_dates, 'nb de pics':list_nb_pics, 'temp moyenne':list_t_moy, 'temp max':list_t_max, 'temp min':list_t_min, 'ecart de temp':list_ecart_t}
df_jour = pd.DataFrame(data = dico_jour)		

#dataframe des pics
peak_dico = {'temps':list_peak_time, 'dbt pic':list_dbt_pic,'fin pic':list_fin_pic, 'hauteur':list_peak_height, 'largeur':list_peak_width, 'aire':list_peak_area_diff, 'pente':list_pente}
df_peaks = pd.DataFrame(data = peak_dico)

#ouverture et écriture des CSV où sont stockées les informations
df_jour.to_csv('descriptif_jours.csv',sep=';')
df_peaks.to_csv('descriptif_pics.csv',sep =';')

#affichage graphique de la température moyenne des jours en fonction du temps
liste_dates_plot = [datetime.date(int(jour[0:4]),int(jour[5:7]),int(jour[8:10])) for jour in list_dates]
plt.plot(liste_dates_plot,list_t_moy)
plt.xlabel('Temps')
plt.ylabel('Température moyenne journalière')
plt.show()