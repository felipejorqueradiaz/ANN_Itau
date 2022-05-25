#%% Librerias
import pandas as pd
import numpy as np
import os
#%% Creación de path ..
print('Antes:',os.getcwd())
path='C:/Users/Asus/Documents/GitHub/ANN_Itau/'
os.chdir(path)
print('desp:',os.getcwd())#os.listdir()

#%% Carga de datos

df_train = pd.read_csv('Datos/raw/Comunicaciones_train.csv')


df_test = pd.read_csv('Datos/raw/Comunicaciones_test.csv')

df_train['dataset'] = 'train'
df_test['dataset'] = 'test'

df = pd.concat([df_train, df_test], ignore_index=True)
df = df.sort_values(['id', 'Periodo'],ascending = [True, True])
df.Periodo = df.Periodo.astype('object') #Lo pasamos a str

#%% Acumulado campañas
a=df.sample(30)
df['id-producto-tipo']=df['id'].astype(str)+"-"+df['Producto-Tipo']
cuenta=df.groupby(['id-producto-tipo'])['dataset'].count().to_frame() #Contar las transacciones

#%% Acumulado campañas con lect

del df_train
del df_test
