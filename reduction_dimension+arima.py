#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:04:57 2020

@author: juliepinol
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pywt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()



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
        coeffs = pywt.wavedec(jour['temperature'], 'db5', 'reflect', level=3)
        matrice_coeffs.append(coeffs[0]) #on ajoute à la matrice la composante basale du signal
        #for i in range(1,len(coeffs)): 
            #matrice_bruit.append(coeffs[i])
        matrice_bruits_cD3.append(coeffs[1])
        matrice_bruits_cD2.append(coeffs[2])
        matrice_bruits_cD1.append(coeffs[3])
    #matrice_coeffs_df = pd.DataFrame(matrice_coeffs)
    #print(index_jour)
    #matrice_coeffs_norm = normalize(matrice_coeffs)
    return matrice_coeffs, matrice_bruits_cD3,matrice_bruits_cD2,matrice_bruits_cD1

### CODE PRINCIPAL

raw_data = pd.read_csv('/Users/juliepinol/Desktop/saumon_ch_positive_complets.csv', sep = ';')
#conversion des jours au format date
raw_data['temps'] = pd.to_datetime(raw_data['temps'])

#remplacement des virgules par des points
raw_data['temperature'] = raw_data['temperature'].str.replace(',','.').apply(float)

	
#remplacement des virgules par des points
if type(raw_data['temperature'][0])!=np.float64:
	raw_data['temperature'] = raw_data['temperature'].str.replace(',','.').apply(float)


	
#création d'une liste de dataframes contenant chacun les données d'une semaine
liste_mois = [group[1] for group in raw_data.groupby(pd.Grouper(key='temps',freq = 'M'))]
#retrait des semaines vides et des semaines ayant moins de 7 jours
liste_mois_clear = []
for mois in liste_mois:
	if mois.shape[0] == 4320 or mois.shape[0] == 4464 :
		liste_mois_clear+=[mois]

liste_mois=liste_mois_clear

#récupération de la transformée en ondelettes
matrice_basale,matrice_bruit1,matrice_bruit2,matrice_bruit3 = ondelettes(liste_mois)


##################### TEST SUR UN MOIS
mois = liste_mois[1]    
#(cA_3, cD_3, cD_2, cD_1) = pywt.wavedec(jour['temperature'], 'db4', 'reflect', level=3)
time_serie = mois['temps']
#print(len(cA_3)+len(cD_1)+len(cD_3)+len(cD_2))
#print(cA_3, cD_3, cD_2, cD_1)
#plt.plot(time_serie,cA_7/10)
#plt.plot(jour['temperature'])
#plt.plot(cD_3)
#plt.plot(cD_2)
#plt.plot(cD_1)
#plt.plot(cA_3)

coeffs = pywt.wavedec(mois['temperature'], 'haar', 'reflect', level=3)

fig = plt.figure(1)
ax = fig.add_subplot(111)
labels = ['cA_2','cD_3', 'cD_2', 'cD_1']
for i in range(len(coeffs)): 
    lab=(labels[i])
    ax.plot(coeffs[i],label = lab)
ax.set_ylabel('Valeur des coefficients de décomposition')
ax.legend(bbox_to_anchor=(1.1, 1.05))

#Courbe du smooth
coeffs_A = coeffs
coeffs_A[1] = np.zeros_like(coeffs_A[1])
coeffs_A[2] = np.zeros_like(coeffs_A[2])
coeffs_A[3] = np.zeros_like(coeffs_A[3])
#print(coeffs_A)
courbe_cA_2 = pywt.waverec(coeffs_A, 'haar')

#Courbe de détail 3
coeffs_D3 = pywt.wavedec(mois['temperature'], 'haar', 'reflect', level=3)
coeffs_D3[0] = np.zeros_like(coeffs_D3[0])
coeffs_D3[2] = np.zeros_like(coeffs_D3[2])
coeffs_D3[-3] = np.zeros_like(coeffs_D3[-3])
#print(coeffs_D3)

courbe_cD_3 = pywt.waverec(coeffs_D3, 'haar')
#Courbe de détail 2
coeffs_D2 = pywt.wavedec(mois['temperature'], 'haar', 'reflect', level=3)
coeffs_D2[0] = np.zeros_like(coeffs_D2[0])
coeffs_D2[1] = np.zeros_like(coeffs_D2[1])
coeffs_D2[-1] = np.zeros_like(coeffs_D2[-1])
courbe_cD_2 = pywt.waverec(coeffs_D2, 'haar')

#Courbe de détail 1
coeffs_D1 = pywt.wavedec(mois['temperature'], 'haar', 'reflect', level=3)
coeffs_D1[0] = np.zeros_like(coeffs_D1[0])
coeffs_D1[1] = np.zeros_like(coeffs_D1[1])
coeffs_D1[2] = np.zeros_like(coeffs_D1[2])
courbe_cD_1 = pywt.waverec(coeffs_D1, 'haar')

coeffs_somme = pywt.wavedec(mois['temperature'], 'haar', 'reflect', level=3)
coeffs_somme[0] = np.zeros_like(coeffs_somme[0])
courbe_somme = pywt.waverec(coeffs_somme, 'haar')

#Plot des différentes niveaux de la transformée
plt.figure(2)
plt.plot(time_serie, courbe_cA_2)
plt.xlabel('Temps')
plt.ylabel('Température lissée')
plt.figure(3)
plt.plot(time_serie, courbe_cD_3)
plt.xlabel('Temps')
plt.ylabel('Température bruits dordre 3')
plt.figure(4)
plt.plot(time_serie, courbe_cD_2)
plt.xlabel('Temps')
plt.ylabel('Température bruits dordre 2')
plt.figure(5)
plt.plot(time_serie, courbe_cD_1)
plt.xlabel('Temps')
plt.ylabel('Température bruits dordre 1')
plt.figure(6)
plt.plot(time_serie, mois['temperature'])
plt.xlabel('Temps')
plt.ylabel('Température initiale capteurs')
plt.figure(7)
plt.plot (time_serie, courbe_somme)
plt.xlabel('Temps')
plt.ylabel('"Bruits" de température')


################## Test ARIMA 
# code modifié et copié à partir de 
# https://machinelearningmastery.com/grid-search-arima-hyperparameters-with-python/
from statsmodels.tsa.arima_model import ARIMA
# contrived dataset
data = courbe_cA_2*100
# fit model
#model = AR(data)
#model_fit = model.fit()
#plt.plot(model_fit)
# make prediction
#yhat = model_fit.predict(len(data), len(data))
#print(yhat)

from sklearn.metrics import mean_squared_error
# split dataset
X = data
train_size = int(len(X) * 0.66)
print(train_size)
train, test = X[0:train_size], X[train_size:]
#print(train)
# train autoregression
model = ARIMA(train,order=(2, 0, 1))
model_fit = model.fit()
#print('Lag: %s' % model_fit.k_ar)
#print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=True)
#for i in range(len(predictions)):
	#print('predicted=%f, expected=%f' % (predictions[i], test[i]))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot results
plt.figure(8)
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()

################## Test hyperparametres ARIMA 
# code modifié et copié à partir de 
# https://machinelearningmastery.com/grid-search-arima-hyperparameters-with-python/

import warnings
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	train_size = int(len(X) * 0.66)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit(disp=0)
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	error = mean_squared_error(test, predictions)
	return error

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					mse = evaluate_arima_model(dataset, order)
					if mse < best_score:
						best_score, best_cfg = mse, order
					print('ARIMA%s MSE=%.3f' % (order,mse))
				except:
					continue
	print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))

# load dataset

p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")

data = courbe_cA_2

evaluate_models(data, p_values, d_values, q_values)
