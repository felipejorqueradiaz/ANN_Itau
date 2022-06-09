import pandas as pd
import numpy as np
import os
import shutil
import copy

periodos=[201901, 201902, 201903, 201904, 201905, 201906, 201907,201908, 201909, 201910, 201911, 201912,
          202001, 202002, 202003,202004, 202005, 202006, 202007,202008, 202009, 202010, 202011] #Para iterar
n=3

#%% Creación de path ..
print('Antes:',os.getcwd())
p = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
path='C:/Users/Asus/Documents/GitHub/ANN_Itau/'
#path = 'C:/Users/Felipe/Documents/Github/ANN_Itau'
os.chdir(path)
print('desp:',os.getcwd())#os.listdir()
from Modelos.functions.utils import bipbop


#%% OBTENER DATASET ORIGINAL, MEZLCAMOS TRAIN Y TEST
#Para crear los .csv
df_train = pd.read_csv('Datos/raw/Transaccion_train.csv', index_col=0) #Hacer un acumulativo de montos o trANSSACCIONES


df_test = pd.read_csv('Datos/raw/Transaccion_test.csv', index_col=0) #Hacer un acumulativo de montos o trANSSACCIONES


df = pd.concat([df_train, df_test], ignore_index=True)
del df_train
del df_test 
df = df.sort_values(['id', 'Periodo'],ascending = [True, True])
df.Periodo = df.Periodo.astype('object') #Lo pasamos a str

del df['Fecha']
del df['Id_Producto']
del df['Tipo']
del df['Signo']
del df['Monto']

ids= df['id'].unique()


df['Target']=1
df.drop_duplicates()

#df.to_csv('Datos/raw/Transaccion_bi.csv',index=False)
bipbop()

#%% Subsetear por periodo:

data={}
for x in df.Periodo.unique():
    data[f"P_{x}"]=df.loc[(df.Periodo ==x)]

del df #Las borramos pq son cuaticas
bipbop()


#%% Definir tipos
def definir_tipos(dataset):    
    dataset.id = dataset.id.astype('object') #Lo pasamos a str
    dataset.Periodo = dataset.Periodo.astype('object') #Lo pasamos a str

dim={}
for i in list(data.keys()):
    dim[i]=data[i].shape[0]
    definir_tipos(data[i])
ncol=data['P_201906'].shape[1]
print(f'El dataset tiene {sum(dim.values())} filas y {ncol} columnas')

bipbop()


#%% Añadimos variables de transacciones por id y producto, de transacciones de id por periodo,

data_enumerate = dict(enumerate(periodos, 1)) #Esto es para enumerar los periodos (para que i-1 sea el periodo anterior)
new_data={}
#Primero hacemos variables base
for i in data_enumerate.keys(): 
    #Variable numerica de cuantas transacciones realizó un id por producto en periodo actual
    data[f"P_{data_enumerate[i]}"].loc[:, ['id-producto-tipo']]=data[f"P_{data_enumerate[i]}"]['id'].astype(str)+"-"+data[f"P_{data_enumerate[i]}"]['Producto-Tipo']
    cuenta=data[f"P_{data_enumerate[i]}"].groupby(['id-producto-tipo'])['Periodo'].count().to_frame() #Contar las transacciones
    cuenta['id-producto-tipo'] = cuenta.index #indice a columna para hacer merge
    cuenta.index.names = ['index'] #le cambiamos el nombre
    cuenta=cuenta.rename(columns={'Periodo': 't_i-0'})#transacciones periodo actual
    new_data[f"P_{data_enumerate[i]}"]=pd.merge(data[f"P_{data_enumerate[i]}"],cuenta,on='id-producto-tipo',how='left')#N° transacciones mes actual
    
    #Agregamos el total de transacciones en el mes en curso
    cuenta2=new_data[f"P_{data_enumerate[i]}"].groupby(['id'])['Periodo'].count().to_frame() #Total de transacciones en el mes en curso
    cuenta2['id'] = cuenta2.index #indice a columna para hacer merge
    cuenta2.index.names = ['index'] #le cambiamos el nombre 
    cuenta2=cuenta2.rename(columns={'Periodo': 'TotalTMes'})#Transacciones totales de esa id ese mes
    new_data[f"P_{data_enumerate[i]}"]=pd.merge(new_data[f"P_{data_enumerate[i]}"],cuenta2,on='id',how='left')#
    
    # Lo pasamos a probabilidades, es decir, dividimos t_0 (transacciones del mes actual) en las transacciones totales
    #Tenemos esta probabilidad en el periodo actual
    new_data[f"P_{data_enumerate[i]}"].loc[:, ['P PT']]=new_data[f"P_{data_enumerate[i]}"]['t_i-0']/new_data[f"P_{data_enumerate[i]}"]['TotalTMes']



bipbop()
#%% Unimos los dataset con las variables base
df=new_data["P_201901"]
for filename in periodos:
    print(filename)
    if filename==201901:
        print('se lo saltó')
    else:
        df=pd.concat([df, new_data[f"P_{filename}"]], ignore_index=True)
        
del new_data
#df.to_pickle('Datos/intermedia/transacciones.pkl', compression= 'zip')
bipbop()

#%% Creamos dataset con periodos e ids para obtener target

b1= pd.DataFrame({'id': np.repeat(ids,len(periodos)), 
                   'Periodo': np.tile(periodos,len(ids))})

bipbop()

#%% Preparación dataset por product-tipo

def preparacion(tipo,diccionario):
    estracto=df[df['Producto-Tipo']==tipo]
    dataset=pd.merge(b1,estracto,on=['id','Periodo'],how='left')#N° transacciones mes anterior
    for x in periodos:
        diccionario[f"P_{x}"]=dataset.loc[(dataset.Periodo ==x)]
        diccionario[f"P_{x}"].loc[:, ['Producto-Tipo']]=tipo
        diccionario[f"P_{x}"].loc[:, ['id-producto-tipo']]=diccionario[f"P_{x}"]['id'].astype(str)+"-"+diccionario[f"P_{x}"]['Producto-Tipo']
    return diccionario
#Para el tipo 'A-A'
AA={}
data_a=preparacion('A-A',AA)
#Para el tipo 'B-B'
BB={}
data_b=preparacion('B-B',BB)
#Para el tipo 'A-A'
CD={}
data_c=preparacion('C-D',CD)
#Para el tipo 'A-A'
DE={}
data_d=preparacion('D-E',DE)
#Para el tipo 'A-A'
EE={}
data_e=preparacion('E-E',EE)

#del df
bipbop()


#%% Lag de la probabildiad

def lag_probabilidad(n,original): #n=1, periodo anterior
    nuevo=copy.deepcopy(original)
    for i in data_enumerate.keys():     
        
        #Y ponemos la probabilidad que tenía el periodo anterior
        if i>=n+1: 
            pp=original[f"P_{data_enumerate[i-n]}"].groupby(['id-producto-tipo'])['P PT'].mean().to_frame() #Todos los id que estaban en el periodo pasado
            pp['id-producto-tipo'] = pp.index #indice a columna para hacer merge
            pp.index.names = ['index'] #le cambiamos el nombre
            pp=pp.rename(columns={'P PT': f'P PT i-{n}'})#probabilidades anteriores
            nuevo[f"P_{data_enumerate[i]}"]=pd.merge(original[f"P_{data_enumerate[i]}"],pp,on='id-producto-tipo',how='left')#Probabilidades mes anterior
        
        else: #Igual creamos la columna pero vacía 
            nuevo[f"P_{data_enumerate[i]}"].loc[:, [f'P PT i-{n}']]=np.nan
        
    return nuevo
        


#n= numero de lag que se quieren
data_a=lag_probabilidad(1,data_a) #1 mes antes
if n>1:
    for i in range(2,n+1):
        data_a=lag_probabilidad(i,data_a) #Desde 2 meses antes
        
        
data_b=lag_probabilidad(1,data_b) #1 mes antes
if n>1:
    for i in range(2,n+1):
        data_b=lag_probabilidad(i,data_b) #Desde 2 meses antes


data_c=lag_probabilidad(1,data_c) #1 mes antes
if n>1:
    for i in range(2,n+1):
        data_c=lag_probabilidad(i,data_c) #Desde 2 meses antes


data_d=lag_probabilidad(1,data_d) #1 mes antes
if n>1:
    for i in range(2,n+1):
        data_d=lag_probabilidad(i,data_d) #Desde 2 meses antes


data_e=lag_probabilidad(1,data_e) #1 mes antes
if n>1:
    for i in range(2,n+1):
        data_e=lag_probabilidad(i,data_e) #Desde 2 meses antes

bipbop()

#%% lag del ntransacciones de id-produc-tipo

#Variable numerica de cuantas transacciones realizó en periodo anterior
# (lag_n_ntrans(j) es la cantidad de transacciones que realizó la id en el producto x 
#en el periodo j anterior)

def lag_n_ntrans(j,dataset): #n=1, periodo anterior
    '''
    j: Orden del lag, ej: j=6, serían las transacciones a de 6 meses antes
    '''

    for i in data_enumerate.keys():     
        
        #Y ponemos la probabilidad que tenía el periodo anterior
        if i>=j+1: 
            pp=dataset[f"P_{data_enumerate[i-j]}"].groupby(['id-producto-tipo'])['t_i-0'].mean().to_frame() #Todos los id que estaban en el periodo pasado
            pp['id-producto-tipo'] = pp.index #indice a columna para hacer merge
            pp.index.names = ['index'] #le cambiamos el nombre
            pp=pp.rename(columns={'t_i-0': f't_i-{j}'})#probabilidades anteriores
            dataset[f"P_{data_enumerate[i]}"]=pd.merge(dataset[f"P_{data_enumerate[i]}"],pp,on='id-producto-tipo',how='left')#Probabilidades mes anterior
        
        else: #Igual creamos la columna pero vacía 
            dataset[f"P_{data_enumerate[i]}"].loc[:, [f't_i-{j}']]=np.nan
        
    return dataset




#n= numero de lag que se quieren
data_a=lag_n_ntrans(1,data_a) #1 mes antes
if n>1:
    for i in range(2,n+1):
        data_a=lag_n_ntrans(i,data_a) #Desde 2 meses antes
        
        
data_b=lag_n_ntrans(1,data_b) #1 mes antes
if n>1:
    for i in range(2,n+1):
        data_b=lag_n_ntrans(i,data_b) #Desde 2 meses antes

data_c=lag_n_ntrans(1,data_c) #1 mes antes
if n>1:
    for i in range(2,n+1):
        data_c=lag_n_ntrans(i,data_c) #Desde 2 meses antes


data_d=lag_n_ntrans(1,data_d) #1 mes antes
if n>1:
    for i in range(2,n+1):
        data_d=lag_n_ntrans(i,data_d) #Desde 2 meses antes


data_e=lag_n_ntrans(1,data_e) #1 mes antes
if n>1:
    for i in range(2,n+1):
        data_e=lag_n_ntrans(i,data_e) #Desde 2 meses antes
        
bipbop()        
        

#%%  Pasamos ese aumento en transacciones a cambio porcentual
def cambio_porcentual_ntrans(j,dataset):
    '''
    j: Es para obtener cuanto, en porcentaje, aumentaron las transacciones del periodo j respecto al periodo j+1.
    '''
    for i in data_enumerate.keys(): 
        dataset[f"P_{data_enumerate[i]}"][f'delta{j}']=(dataset[f"P_{data_enumerate[i]}"][f't_i-{j}']-dataset[f"P_{data_enumerate[i]}"][f't_i-{j+1}'])/dataset[f"P_{data_enumerate[i]}"][f't_i-{j}']
    return dataset



#n= numero de lag que se quieren
data_a=cambio_porcentual_ntrans(1,data_a) #1 mes antes
if n>1:
    for i in range(2,n):
        data_a=cambio_porcentual_ntrans(i,data_a) #Desde 2 meses antes
        
        
data_b=cambio_porcentual_ntrans(1,data_b) #1 mes antes
if n>1:
    for i in range(2,n):
        data_b=cambio_porcentual_ntrans(i,data_b) #Desde 2 meses antes

data_c=cambio_porcentual_ntrans(1,data_c) #1 mes antes
if n>1:
    for i in range(2,n):
        data_c=cambio_porcentual_ntrans(i,data_c) #Desde 2 meses antes


data_d=cambio_porcentual_ntrans(1,data_d) #1 mes antes
if n>1:
    for i in range(2,n):
        data_d=cambio_porcentual_ntrans(i,data_d) #Desde 2 meses antes


data_e=cambio_porcentual_ntrans(1,data_e) #1 mes antes
if n>1:
    for i in range(2,n):
        data_e=cambio_porcentual_ntrans(i,data_e) #Desde 2 meses antes
        
bipbop() 

#%% Creación de target y separación por tipo de producto

aa=data_a["P_201901"]
for filename in periodos:
    print(filename)
    if filename==201901:
        print('se lo saltó')
    else:
        aa=pd.concat([aa, data_a[f"P_{filename}"]], ignore_index=True)
  
del aa['Producto-Tipo']
del aa['id-producto-tipo']
aa.to_pickle('Datos/intermedia/base_tAA.pkl', compression= 'zip')



bb=data_b["P_201901"]
for filename in periodos:
    print(filename)
    if filename==201901:
        print('se lo saltó')
    else:
        bb=pd.concat([bb, data_b[f"P_{filename}"]], ignore_index=True)
del bb['Producto-Tipo']
del bb['id-producto-tipo']
bb.to_pickle('Datos/intermedia/base_tBB.pkl', compression= 'zip')



cc=data_c["P_201901"]
for filename in periodos:
    print(filename)
    if filename==201901:
        print('se lo saltó')
    else:
        cc=pd.concat([cc, data_c[f"P_{filename}"]], ignore_index=True)
del cc['Producto-Tipo']
del cc['id-producto-tipo']
cc.to_pickle('Datos/intermedia/base_tCD.pkl', compression= 'zip')

dd=data_d["P_201901"]
for filename in periodos:
    print(filename)
    if filename==201901:
        print('se lo saltó')
    else:
        dd=pd.concat([dd, data_d[f"P_{filename}"]], ignore_index=True)
del dd['Producto-Tipo']
del dd['id-producto-tipo']
dd.to_pickle('Datos/intermedia/base_tDE.pkl', compression= 'zip')

ee=data_e["P_201901"]
for filename in periodos:
    print(filename)
    if filename==201901:
        print('se lo saltó')
    else:
        ee=pd.concat([ee, data_e[f"P_{filename}"]], ignore_index=True)
        
del ee['Producto-Tipo']
del ee['id-producto-tipo']
ee.to_pickle('Datos/intermedia/base_tEE.pkl', compression= 'zip')


bipbop()