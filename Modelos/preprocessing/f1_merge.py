import os 
import pandas as pd
import pickle
 

#%% Carga de dataset
path='C:/Users/Asus/Documents/GitHub/ANN_Itau'

os.chdir(path)

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


