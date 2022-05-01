import pandas as pd
import numpy as np
import os
import shutil


periodos=[201901, 201902, 201903, 201904, 201905, 201906, 201907,201908, 201909, 201910, 201911, 201912,
          202001, 202002, 202003,202004, 202005, 202006, 202007,202008, 202009, 202010, 202011] #Para iterar


#%% Creación de path ..
print('Antes:',os.getcwd())
# p = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
# os.chdir(path)
# print('desp:',os.getcwd())#os.listdir()

#%% OBTENER DATASET ORIGINAL, MEZLCAMOS TRAIN Y TEST
#Para crear los .csv
df_train = pd.read_csv('Datos/raw/Transaccion_train.csv', index_col=0) #Hacer un acumulativo de montos o trANSSACCIONES
df_train['dataset'] = 'train'

df_test = pd.read_csv('Datos/raw/Transaccion_test.csv', index_col=0) #Hacer un acumulativo de montos o trANSSACCIONES
df_test['dataset'] = 'test'

df = pd.concat([df_train, df_test], ignore_index=True)

df = df.sort_values(['id', 'Periodo'],ascending = [True, True])
df.Periodo = df.Periodo.astype('object') #Lo pasamos a str

#%% Subsetear por periodo:
if (not os.path.exists('Datos/raw/transaction_subset')):# Crear carpeta ./transaction_subset
    os.mkdir('Datos/raw/transaction_subset')
else:# Eliminar anteriores para reemplazarlos:
    shutil.rmtree('Datos/raw/transaction_subset')
    os.mkdir('Datos/raw/transaction_subset')
data={}
for x in df.Periodo.unique():
    data[f"P_{x}"]=df.loc[(df.Periodo ==x)]
    #print(f"df_Periodo{x}",globals()[f"df_Periodo{x}"].describe())
    data[f"P_{x}"].to_csv('Datos/raw/transaction_subset/'+f"P_{x}"+'.csv',index=False)
del df #Las borramos pq son cuaticas
del df_train
del df_test

# data = {}# Para leer los .csv
# size={}
# for filename in periodos:
#     data[f"P_{filename}"] = pd.read_csv('Datos/raw/transaction_subset/'+f"P_{filename}"+'.csv')
#     size[f"P_{filename}"]=data[f"P_{filename}"].shape[0]

#%%
def definir_tipos(dataset):    
    dataset.id = dataset.id.astype('object') #Lo pasamos a str
    dataset.Signo = dataset.Signo.astype('object') #tb
    dataset.loc[:, 'Fecha'] =pd.to_datetime(dataset.Fecha) #A fecha
    dataset.Periodo = dataset.Periodo.astype('object') #Lo pasamos a str

dim={}
for i in list(data.keys()):
    dim[i]=data[i].shape[0]
    definir_tipos(data[i])
ncol=data['P_201906'].shape[1]
print(f'El dataset tiene {sum(dim.values())} filas y {ncol} columnas')

#%% Añadimos variables de transacciones por id y producto, de transacciones de id por periodo,
# y 
data_enumerate = dict(enumerate(periodos, 1)) #Esto es para enumerar los periodos (para que i-1 sea el periodo anterior)
new_data={}
#Primero hacemos variables base
for i in data_enumerate.keys(): 
    #Variable numerica de cuantas transacciones realizó un id por producto en periodo actual
    data[f"P_{data_enumerate[i]}"].loc[:, ['id-producto']]=data[f"P_{data_enumerate[i]}"]['id'].astype(str)+"-"+data[f"P_{data_enumerate[i]}"]['Id_Producto']
    cuenta=data[f"P_{data_enumerate[i]}"].groupby(['id-producto'])['Monto'].count().to_frame() #Contar las transacciones
    cuenta['id-producto'] = cuenta.index #indice a columna para hacer merge
    cuenta.index.names = ['index'] #le cambiamos el nombre
    cuenta=cuenta.rename(columns={'Monto': 't_i-0'})#transacciones periodo actual
    new_data[f"P_{data_enumerate[i]}"]=pd.merge(data[f"P_{data_enumerate[i]}"],cuenta,on='id-producto',how='left')#N° transacciones mes actual
    
    #Agregamos el total de transacciones en el mes en curso
    cuenta2=new_data[f"P_{data_enumerate[i]}"].groupby(['id'])['Monto'].count().to_frame() #Total de transacciones en el mes en curso
    cuenta2['id'] = cuenta2.index #indice a columna para hacer merge
    cuenta2.index.names = ['index'] #le cambiamos el nombre 
    cuenta2=cuenta2.rename(columns={'Monto': 'TotalTMes'})#Transacciones totales de esa id ese mes
    new_data[f"P_{data_enumerate[i]}"]=pd.merge(new_data[f"P_{data_enumerate[i]}"],cuenta2,on='id',how='left')#
    
    # Lo pasamos a probabilidades, es decir, dividimos t_0 (transacciones del mes actual) en las transacciones totales
    #Tenemos esta probabilidad en el periodo actual
    new_data[f"P_{data_enumerate[i]}"]['P Producto']=new_data[f"P_{data_enumerate[i]}"]['t_i-0']/new_data[f"P_{data_enumerate[i]}"]['TotalTMes']
    #Y ponemos la probabilidad que tenía el periodo anterior
    if i>=2:    
        new_data[f"P_{data_enumerate[i]}"]['P Producto i-1']=new_data[f"P_{data_enumerate[i-1]}"]['t_i-0']/new_data[f"P_{data_enumerate[i-1]}"]['TotalTMes']
    else: #Igual creamos la columna pero vacía 
        new_data[f"P_{data_enumerate[i]}"]['P Producto i-1']=np.nan
        
#%%

#Variable numerica de cuantas transacciones realizó en periodo anterior
# (lag_n_ntrans(j) es la cantidad de transacciones que realizó la id en el producto x 
#en el periodo j anterior)
def lag_n_ntrans(j):
    '''
    j: Orden del lag, ej: j=6, serían las transacciones a de 6 meses antes
    '''
    for i in data_enumerate.keys(): #Variable numerica de cuantas transacciones realizó en periodo anterior
        if i>=(j+1):
            cuenta=data[f"P_{data_enumerate[i-j]}"].groupby(['id-producto'])['Monto'].count().to_frame() #Contar las transacciones periodo anterior de la persona y producto
            cuenta['id-producto'] = cuenta.index #indice a columna para hacer merge
            cuenta.index.names = ['index'] #le cambiamos el nombre
            cuenta=cuenta.rename(columns={'Monto': f't_i-{j}'})#transacciones anteriores
            new_data[f"P_{data_enumerate[i]}"]=pd.merge(new_data[f"P_{data_enumerate[i]}"],cuenta,on='id-producto',how='left')#N° transacciones mes anterior
        else:
            new_data[f"P_{data_enumerate[i]}"][f't_i-{j}']=np.nan #Igualmente creamos columna, pero vacía para periodos menores a j
            
lag_n_ntrans(1) #1 mes antes
lag_n_ntrans(2) #2 meses antes
lag_n_ntrans(3) #3 meses antes
#%%  Pasamos ese aumento en transacciones a cambio porcentual
def cambio_porcentual_ntrans(j):
    '''
    j: Es para obtener cuanto, en porcentaje, aumentaron las transacciones del periodo j respecto al periodo j+1.
    '''
    for i in data_enumerate.keys(): 
        new_data[f"P_{data_enumerate[i]}"][f'delta{j}']=(new_data[f"P_{data_enumerate[i]}"][f't_i-{j}']-new_data[f"P_{data_enumerate[i]}"][f't_i-{j+1}'])/new_data[f"P_{data_enumerate[i]}"][f't_i-{j}']

cambio_porcentual_ntrans(0) #aumento de transacciones del periodo 0 respecto al periodo 1.
cambio_porcentual_ntrans(1)
cambio_porcentual_ntrans(2)


#%% Variable de si id tenía producto en periodo anterior

#Caso base i=2

new_data['P_201901']['Tenia Producto']=np.nan

i=2
tenia=new_data[f"P_{data_enumerate[i-1]}"].groupby(['id-producto'])['Monto'].count().to_frame() #Todos los id que estaban en el periodo pasado
tenia['Monto']=1
tenia['id-producto'] = tenia.index #indice a columna para hacer merge
tenia.index.names = ['index'] #le cambiamos el nombre
tenia=tenia.rename(columns={'Monto': 'Tenia Producto'})#transacciones anteriores
new_data[f"P_{data_enumerate[i]}"]=pd.merge(new_data[f"P_{data_enumerate[i]}"],tenia,on='id-producto',how='left')#N° transacciones mes anterior

#Generalizamos

for i in data_enumerate.keys(): 
    if i>=3:
        tenia2=new_data[f"P_{data_enumerate[i-1]}"].groupby(['id-producto'])['Monto'].count().to_frame() #Todos los id que estaban en el periodo pasado
        tenia2['Monto']=1
        tenia2['id-producto'] = tenia2.index #indice a columna para hacer merge
        tenia2.index.names = ['index'] #le cambiamos el nombre
        tenia2=tenia2.rename(columns={'Monto': 'Tenia Producto'})#transacciones anteriores
        tenia_diff = tenia2[~tenia2['id-producto'].isin(tenia['id-producto'])]
        tenia = pd.concat([tenia, tenia_diff])
        new_data[f"P_{data_enumerate[i]}"]=pd.merge(new_data[f"P_{data_enumerate[i]}"],tenia,on='id-producto',how='left')#N° transacciones mes anterior
        
        
#%% Signo a numero
for i in data_enumerate.keys(): 
    new_data[f"P_{data_enumerate[i]}"]['Signo']=new_data[f"P_{data_enumerate[i]}"]['Signo'].replace(['Positivo','Negativo',np.nan],[1,-1,0])
    
#%%  Guardar .csv
df=new_data["P_201901"]
for filename in periodos:
    print(filename)
    if filename==201901:
        print('se lo saltó')
    else:
        df=pd.concat([df, new_data[f"P_{filename}"]], ignore_index=True)

df.to_csv('Datos/transacciones_v1.csv',index=False)