# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 13:11:30 2022

@author: Asus
"""
import pandas as pd
import os

#%% Creación de path ..
path='C:/Users/Asus/Documents/GitHub/ANN_Itau/'
#path = 'C:/Users/Felipe/Documents/Github/ANN_Itau'
os.chdir(path)
from Modelos.functions.utils import bipbop

#%% OBTENER DATASET ORIGINAL, MEZLCAMOS TRAIN Y TEST
#Para crear los .csv
df_train = pd.read_csv('Datos/raw/Transaccion_train.csv', index_col=0) #Hacer un acumulativo de montos o trANSSACCIONES
df_test = pd.read_csv('Datos/raw/Transaccion_test.csv', index_col=0) #Hacer un acumulativo de montos o trANSSACCIONES


df = pd.concat([df_train, df_test], ignore_index=True)

#%% Partimos con el análisis de los datos:
print(df.sample(5))
print(df.describe().apply(lambda s: s.apply('{0:.5f}'.format)))
print(df.describe(include=['object']))
print(df.isnull().sum())
#%% Id


aaa=df[df.Monto<0] 
#%% Producto 

#%% Tipo

#%% Fecha




#%% Monto
negativos=df[df.Monto<0]

#%% Signo

#%% Periodos




