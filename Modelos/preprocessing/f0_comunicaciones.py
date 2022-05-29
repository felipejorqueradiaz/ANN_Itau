#%% Librerias
import pandas as pd
import numpy as np
import os
#%% Creación de path ..
print('Antes:',os.getcwd())
#path='C:/Users/Asus/Documents/GitHub/ANN_Itau/'
path = 'C:/Users/Felipe/Documents/Github/ANN_Itau'
os.chdir(path)
print('desp:',os.getcwd())#os.listdir()

#%% Carga de datos

df_train = pd.read_csv('Datos/raw/Comunicaciones_train.csv',
                       index_col=0)


df_test = pd.read_csv('Datos/raw/Comunicaciones_test.csv',
                      index_col=0)

df_train['dataset'] = 'train'
df_test['dataset'] = 'test'

df = pd.concat([df_train, df_test], ignore_index=True)
df = df.sort_values(['id', 'Periodo'],ascending = [True, True])
#df.Periodo = df.Periodo.astype('object') #Lo pasamos a str

del df_train
del df_test
#%% Acumulado comunicaciones
df.drop(['Fecha', 'Tipo_comunicacion'], axis=1, inplace=True)
df = df.groupby(['id', 'Periodo','Id_Producto', 'Tipo', 'Producto-Tipo'],
                as_index=False).agg({'Lectura': ['sum','count']})

df.columns = ['id', 'Periodo', 'Id_Producto', 'Tipo',
              'Producto-Tipo', 'N_Lecturas', 'N_comunicaciones']

df['Efic_comunicacion'] = df['N_Lecturas'] / df['N_comunicaciones']
#%% Acumulado comunicaciones con lect
df.sort_values(by=['id', 'Producto-Tipo', 'Periodo'], inplace=True)
df.reset_index(drop=True, inplace=True)

df['Com_cumsum'] = df.groupby(['id',
                               'Producto-Tipo']
                              )['N_comunicaciones'].transform(pd.Series.cumsum)

df['Efic_cumsum'] = df.groupby(['id',
                                'Producto-Tipo']
                               )['N_Lecturas'].transform(pd.Series.cumsum)

df['Efic_historica'] = df['Efic_cumsum']/df['Com_cumsum']

#%%
df.to_pickle('Datos/intermedia/comunicaciones.pkl', compression= 'zip')