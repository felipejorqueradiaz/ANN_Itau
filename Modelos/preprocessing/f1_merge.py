import os 
import pandas as pd
import pickle
 

#%% Carga de dataset
path='C:/Users/Asus/Documents/GitHub/ANN_Itau'

os.chdir(path)


#%%

trans_A = pd.read_pickle('Datos/intermedia/base_tAA.pkl', compression= 'zip')

#%%

campanas = pd.read_pickle('Datos/intermedia/campañas.pkl', compression= 'zip')
campanas.drop(['dataset', 'Id_Producto', 'Tipo'], axis=1, inplace=True)

comunicaciones = pd.read_pickle('Datos/intermedia/comunicaciones.pkl', compression= 'zip')
comunicaciones.drop(['Id_Producto', 'Tipo'], axis=1, inplace=True)


#%%

base_A = trans_A.merge(campanas[campanas['Producto-Tipo'] == 'A-A'].drop('Producto-Tipo', axis=1), how='left')


#%%

#%%


# transacciones_previa.to_pickle('Datos/intermedia/transacciones.pkl', compression= 'bz2')

transacciones= pd.read_pickle('Datos/intermedia/transacciones.pkl', compression= 'zip')
# campanas=pd.read_csv('Datos/intermedia/campañas.csv')
# 
#Faltan:
#comunicaciones=pd.read_csv('Datos/intermedia/comunicaciones.plk', compression= 'bz2')
#comunicaciones=pd.read_csv('Datos/intermedia/consumidores.plk', compression= 'bz2')

#%%



#%%



#%%
out = pd.DataFrame(ids.merge(pd.DataFrame(periodos),how='cross')

#%%





data = {'id': ['Tom', 'Joseph', 'Krish', 'John'], 'Age': [20, 21, 19, 18]}  
  
# Create DataFrame  
df = pd.DataFrame(data) 

#%% Merge

u1=pd.merge(transacciones,campanas, on=['id', 'Id_Producto', 'Tipo', 'Producto-Tipo','dataset', 'Periodo'],how='left')



#%% tipo de variables
u1['id']=u1['id'].astype(str)

#%%




#%%
print(transacciones.describe())






#%%

u2.to_pickle('Datos/intermedia/union.pkl', compression= 'bz2')

#%%

a=transacciones.sample(100)


