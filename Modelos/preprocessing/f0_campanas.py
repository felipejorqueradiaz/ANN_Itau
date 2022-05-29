import pandas as pd
import numpy as np
import os

#%% Carga de Datos
print(os.getcwd())
path = 'C:/Users/Felipe/Documents/Github/ANN_Itau'
#path='C:/Users/Asus/Documents/GitHub/ANN_Itau'

os.chdir(path)

from Modelos.functions.utils import bipbop
#%%
df_train = pd.read_csv('Datos/raw/Campanas_train.csv',
                       index_col=0)
df_train['dataset'] = 'train'

df_test = pd.read_csv('Datos/raw/Campanas_test.csv',
                      index_col=0)
df_test['dataset'] = 'test'


df = pd.concat([df_train, df_test], axis=0)
df.drop('Fecha_Campaña', axis=1, inplace=True)

del df_train
del df_test

bipbop()
#%%

#Buscamos aquellas personas con campaña y mayor a 1 mes
df_no1 = df[df['Duracion_Campaña'] != 1].reset_index(drop=True)

#Repetimos la fila
df_no1 = df_no1.assign(Times = df_no1['Duracion_Campaña'].apply(lambda x: range(0, x))).explode('Times')
#Obtenemos el mes
df_no1['Mes'] = df_no1['Periodo']%100
#Obtenemos el año
df_no1['Año'] = (df_no1['Periodo']-df_no1['Mes'])/100
#Propagamos la campaña
df_no1['Mes'] = df_no1['Mes']+df_no1['Times']
df_no1.loc[df_no1['Mes']>=13,'Año'] = df_no1[df_no1['Mes']>=13]['Año']+1
df_no1.loc[df_no1['Mes']>=13,'Mes'] = df_no1[df_no1['Mes']>=13]['Mes']-12
df_no1['Periodo'] = df_no1['Año']*100 + df_no1['Mes']
df_no1.drop(['Mes', 'Año', 'Times'], axis=1, inplace=True)

df = df[df['Duracion_Campaña'] == 1]
df = pd.concat([df, df_no1], axis=0).reset_index(drop=True)

#%%

#Re-creamos la variable resultado
df.loc[df['Periodo']>=202008, 'dataset'] = 'test'
df.loc[df['Periodo']>=202008, 'Resultado'] = np.nan

#Creamos variable dummy
df.rename(columns={"Duracion_Campaña": "tiene_camp"}, inplace=True)
df.drop(['tiene_camp'], axis=1, inplace=True)

#Rellenamos los meses faltantes sin campaña
#df['Date'] = pd.to_datetime(df.Periodo.astype(int).astype(str),format='%Y%m')
df.drop_duplicates(subset = ['id', 'Id_Producto', 'Tipo', 'Producto-Tipo', 'Canal'], inplace=True)
df.reset_index(drop=True, inplace=True)

#%% 6 new

df_B = df[df['Canal'] == 'B']
df_B['Canal_B'] = 1
df_B.drop('Canal', axis = 1, inplace=True)

df_C = df[df['Canal'] == 'C']
df_C['Canal_C'] = 1
df_C.drop('Canal', axis = 1, inplace=True)

df = df_B.merge(df_C, how = 'outer', on = ['id', 'Id_Producto', 'Tipo',
                                            'Producto-Tipo', 'Periodo',
                                            'dataset'])
df.fillna(0, inplace=True)
#%%

df['Resultado'] = df[['Resultado_x', 'Resultado_y']].max(axis=1)
df.drop(['Resultado_x', 'Resultado_y'], axis=1, inplace=True)
#%% 6
df.sort_values(by=['id', 'Producto-Tipo', 'Periodo'], inplace=True)
df.reset_index(drop=True, inplace=True)

df['CanalPT_C_cumsum'] = df.groupby(['id', 'Producto-Tipo'])['Canal_C'].transform(pd.Series.cumsum)
df['CanalPT_B_cumsum'] = df.groupby(['id', 'Producto-Tipo'])['Canal_B'].transform(pd.Series.cumsum)
df['Camp_PT_cumsum'] = df['CanalPT_C_cumsum'] + df['CanalPT_B_cumsum']

bipbop()
#%%
df['Canal_C_cumsum'] = df.groupby(['id'])['Canal_C'].transform(pd.Series.cumsum)
df['Canal_B_cumsum'] = df.groupby(['id'])['Canal_B'].transform(pd.Series.cumsum)
df['Camp_cumsum'] = df['Canal_C_cumsum'] + df['Canal_B_cumsum']

bipbop()
#%% 7
#df['Periodo'] = df.Periodo.astype(int)
#%% 8
df.loc[df['Periodo']>=202008, 'dataset'] = 'test'
df.loc[df['Periodo']<202008, 'dataset'] = 'train'

#%% 9
# df.to_csv('Datos/intermedia/campañas.csv', index=False)

df.to_pickle('Datos/intermedia/campañas.pkl', compression= 'zip')
bipbop()
