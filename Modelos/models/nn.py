import os
import pandas as pd
import numpy as np

from imblearn.under_sampling import RandomUnderSampler #Undersampling
from sklearn.metrics import classification_report
import ml_metrics as metrics
# In order to ignore FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
## Ahora se calcula la matriz de confusion
from keras.callbacks import EarlyStopping
########################################
from tensorflow.keras.optimizers import Adam



#%% Carga de dataset
#%% Creación de path ..

#### PONER EL PATH A LA CARPETA MADRE ACÁ

#path='C:/Users/Asus/Documents/GitHub/ANN_Itau/'
#path = 'C:/Users/Felipe/Documents/Github/ANN_Itau'
os.chdir(path)


product_list = ['A-A',
                'B-B',
                'C-D',
                'D-E',
                'E-E']



target= pd.read_pickle('Datos/final/Target.pkl', compression= 'zip')

#%% Lectura de Train/Test

train = {}
test = {}
val = {}

for prod in product_list:
    train[prod] = pd.read_pickle('Datos/final/{}_train.pkl'.format(prod), compression= 'zip')
    test[prod] = pd.read_pickle('Datos/final/{}_test.pkl'.format(prod), compression= 'zip')
    val[prod] = pd.read_pickle('Datos/final/{}_val.pkl'.format(prod), compression= 'zip')


#%% Creación de red
#Se pobará con 3 distintos tipos de modelos:
# Modelo1: 10, 8 ,8
# Modelo2:4,3,3
# Modelo 3: 5 5

def MiRed(n_features):
    model = Sequential()

    ## Adicionalmente, una capa de tipo 'Dense' es como lo que hemos visto en clase
    ## Donde TODAS las neuronas de una capa se conectan con TODAS las neuronas de la siguiente capa
    model.add(Dense(10, activation='relu',input_shape=(n_features,)))
    ## Esta es la segunda capa, tiene 100 neuronas, pero no es necesario especificar cuantas columnas entran.
    model.add(Dense(8, activation='relu')) 
    
    model.add(Dense(8, activation='relu')) 

    
    ## Finalmente la capa de salida tiene solo una neurona, sin FA.
    model.add(Dense(1))
    
    return model


#%% Ahora vamos a entrenar y guardar las redes, con distintas tasas de aprendizaje


lr_to_test = [ 0.1]  #tres tasas de aprendizaje diferentes, una pequeña, mediana y grande
#0.01,  10

for i in lr_to_test:
    for prod in product_list:
        rus = RandomUnderSampler(random_state=0)
    
        X = train[prod].drop(['id', 'Periodo', 'Target'], axis=1)
        y = train[prod]['Target']
        
        X_train_us, y_train_us = rus.fit_resample(X, y)
        
        X_test = test[prod].drop(['id', 'Periodo', 'Target'], axis=1)
        y_test = test[prod]['Target']
        id_per = test[prod][['id', 'Periodo']]
    
        print(f"Modelo {prod} lr {i}\n")
        n_features = X.shape[1]
        cb =EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5)
        model1 = MiRed(n_features)
        model1.summary()
        adam = Adam()
        model1.compile(optimizer= adam,loss='mean_squared_error', metrics=['accuracy'])
        model1.fit(X_train_us,y_train_us
                   , epochs=100
                   , verbose=1,
                   callbacks=[cb]
                   )

        path = f'./modelo1088{prod}Adam.h5'
        model1.save(path)
    
#%% Predecimos variables, las unicmos en un dataset y obtenemos métricas
real = pd.DataFrame()
pred = pd.DataFrame()
valid = pd.DataFrame()                    
### Genero la predicción con el modelo

for prod in product_list:
    rus = RandomUnderSampler(random_state=0)
    
    X = train[prod].drop(['id', 'Periodo', 'Target'], axis=1)
    y = train[prod]['Target']
        
    X_train_us, y_train_us = rus.fit_resample(X, y)
        
    X_test = test[prod].drop(['id', 'Periodo', 'Target'], axis=1)
    y_test = test[prod]['Target']
    id_per = test[prod][['id', 'Periodo']]
    # file_to_read = open(f"./modelo1{prod}.pickle", "rb")
    # model1 = pickle.load(file_to_read)
    # file_to_read.close()
    model1 = keras.models.load_model(f'./modelo1088{prod}Adam.h5')


    y_pred1 = model1.predict(X_test) 
    ya=y_pred1

    pred[prod] = y_pred1.tolist()
    pred[prod] = pred[prod].str[0]
    real[prod] = y_test
    
    #VALIDACION
    X_val = val[prod].drop(['id', 'Periodo', 'Target'], axis=1)
    id_per_val = val[prod][['id']]
    valid[prod] = model1.predict(X_val).tolist()  
    valid[prod] = valid[prod].str[0]
    
    y_pred1[y_pred1<0.5]=0
    y_pred1[y_pred1>=0.5]=1
    print('-----------\nPRODUCTO {}\n'.format(prod),classification_report(y_test, y_pred1),'\n\n')

#%% Añadimos id y periodo a los resultados
real = pd.concat([real, id_per], axis = 1, ignore_index=True)
pred = pd.concat([pred, id_per.reset_index(drop = True)], axis = 1, ignore_index=True)
valid = pd.concat([valid, id_per_val.reset_index(drop = True)], axis = 1, ignore_index=True)

real.columns = product_list + ['id', 'Periodo']
pred.columns = product_list + ['id', 'Periodo']
valid.columns = product_list + ['id']


#%% Obtenemos el mapk según distintos puntos de corte

prod_vector = np.array(product_list)
cortes = np.arange(0,1.01,0.05)

values = []
for corte in cortes:
    for mes in pred.Periodo.unique():
        real_temp = target[target['Periodo'] == mes].reset_index(drop=True)
        list_true = real_temp['productos'].to_numpy(copy = True).tolist()
        
     
        pred_temp = pred[pred['Periodo'] == mes]
        d_pred = pred_temp[product_list].to_numpy(copy = True)
        pred_sort_mask = d_pred.argsort()
        d2_pred = np.where(d_pred <= corte, 0, 1)
        v_pred = np.where(d2_pred, prod_vector, 'nulo')
        v_pred = np.take_along_axis(v_pred,pred_sort_mask,axis=1)[:, [4, 3, 2, 1, 0]].tolist()
        v_pred = [[valor for valor in lista if valor!='nulo'] for lista in v_pred]
        valor = metrics.mapk(list_true, v_pred, 5)
        print('EL MAP5 para el mes {} es:'.format(mes), valor)
        values.append([corte, mes, valor])
    
    
    list_true = target.reset_index(drop=True)['productos'].to_numpy(copy = True).tolist()
    
    pred_temp = pred.copy()
    d_pred = pred_temp[product_list].to_numpy(copy = True)
    pred_sort_mask = d_pred.argsort()
    d2_pred = np.where(d_pred <= corte, 0, 1)
    v_pred = np.where(d2_pred, prod_vector, 'nulo')
    v_pred = np.take_along_axis(v_pred,pred_sort_mask,axis=1)[:, [4, 3, 2, 1, 0]].tolist()
    v_pred = [[valor for valor in lista if valor!='nulo'] for lista in v_pred]
    valor = metrics.mapk(list_true, v_pred, 5)
    values.append([corte, 'general', valor])
    print('EL MAP5 en general es:', valor)

values = pd.DataFrame(values)
values.columns = ['corte', 'periodo', 'map5']
#%% plot
import seaborn as sns
plot=sns.lineplot(data=values, x="corte", y="map5", hue="periodo")
#%% f1 El presume que el punto de corte más representativo es 0.5

corte = 0.5

d_val = valid[product_list].to_numpy(copy = True)
val_sort_mask = d_val.argsort()
d2_val = np.where(d_val <= corte, 0, 1)

v_val = np.where(d2_val, prod_vector, 'nulo')
v_val = np.take_along_axis(v_val,val_sort_mask,axis=1)[:, [4, 3, 2, 1, 0]].tolist()
v_val = [[valor for valor in lista if valor!='nulo'] for lista in v_val]
final = [' '.join(row).strip() for row in v_val]

#%% Guardamos resultado de modelo

pred_final = pd.DataFrame()
pred_final['id']=valid['id'].copy()
pred_final['productos'] = final

pred_final['id']=pred_final['id'].astype(np.int64)
pred_final=pred_final.fillna(" ")
pred_final.to_csv('Resultados/redneuronal1088adam.csv',index=False)
#%%


    

