#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 14:43:36 2020

@author: juliepinol
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


raw_data = pd.read_csv('/Users/juliepinol/Desktop/Test_cluster_saumon_epures_norm.csv', sep = ';')
#raw_data['temp moyenne'] = raw_data['temp moyenne'].str.replace(',','.').apply(float)

#raw_data['cluster'] = raw_data['cluster'].apply(int)

#affichage graphique de la température moyenne des jours en fonction du temps
#colors = {'1':'red', '2':'blue', '3':"yellow",'4':'black'}
colormap = np.array(['w','r', 'g', 'b','y','c','k','m'])
print(colormap[raw_data['cluster']])
plt.scatter(raw_data['day'],raw_data['temp moyenne'],color= colormap[raw_data['cluster']])
plt.xlabel('Temps')
plt.ylabel('Température moyenne journalière')
plt.show()

