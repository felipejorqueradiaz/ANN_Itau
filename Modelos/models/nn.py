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

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
## Ahora se calcula la matriz de confusion
from keras.callbacks import EarlyStopping
########################################
from tensorflow import keras

#%% Carga de dataset
path='C:/Users/Asus/Documents/GitHub/ANN_Itau'
#path = 'C:/Users/Felipe/Documents/Github/ANN_Itau'
product_list = ['A-A',
                'B-B',
                'C-D',
                'D-E',
                'E-E']

os.chdir(path)

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


#%% Entrenamiento de modelos


for prod in product_list:
    rus = RandomUnderSampler(random_state=0)
    
    X = train[prod].drop(['id', 'Periodo', 'Target'], axis=1)
    y = train[prod]['Target']
        
    X_train_us, y_train_us = rus.fit_resample(X, y)
        
    X_test = test[prod].drop(['id', 'Periodo', 'Target'], axis=1)
    y_test = test[prod]['Target']
    id_per = test[prod][['id', 'Periodo']]
    
    print(f"Modelo {prod}\n")
    n_features = X.shape[1]
    cb =EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5)
    print("Modelo 1\n")
    model1 = MiRed(n_features)
    model1.summary()
    model1.compile(optimizer='adam',loss='mean_squared_error', metrics=['accuracy'])
    model1.fit(X_train_us,y_train_us
          , epochs=100
          , verbose=1,
          callbacks=[cb]
          )
    model1.save(f'/Modelos/models/modelo1{prod}.h5')
    
#%%       
real = pd.DataFrame()
pred = pd.DataFrame()
valid = pd.DataFrame()                    
### Genero la predicción con el modelo
for prod in product_list:
    model1 = keras.models.load_model('f/Modelos/models/modelo1{prod}.h5')
    y_pred1 = model1.predict(X_test) 
    ya=y_pred1

    pred[prod] = y_pred1.tolist()
    pred[prod] = pred[prod].str[0]
    
    y_pred1[y_pred1<0.5]=0
    y_pred1[y_pred1>=0.5]=1
    print('-----------\nPRODUCTO {}\n'.format(prod),classification_report(y_test, y_pred1),'\n\n')
    real[prod] = y_test
    
    #VALIDACION
    X_val = val[prod].drop(['id', 'Periodo', 'Target'], axis=1)
    id_per_val = val[prod][['id']]
    valid[prod] = model1.predict(X_test).tolist()   #trasponer?
    
    
#%%
real = pd.concat([real, id_per], axis = 1, ignore_index=True)
pred = pd.concat([pred, id_per.reset_index(drop = True)], axis = 1, ignore_index=True)
valid = pd.concat([valid, id_per_val.reset_index(drop = True)], axis = 1, ignore_index=True)

real.columns = product_list + ['id', 'Periodo']
pred.columns = product_list + ['id', 'Periodo']
valid.columns = product_list + ['id']

#%%

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
#%%
import seaborn as sns
plot=sns.lineplot(data=values, x="corte", y="map5", hue="periodo")
#%%

corte = 0.5

d_val = valid[product_list].to_numpy(copy = True)
val_sort_mask = d_val.argsort()
d2_val = np.where(d_val <= corte, 0, 1)

v_val = np.where(d2_val, prod_vector, 'nulo')
v_val = np.take_along_axis(v_val,val_sort_mask,axis=1)[:, [4, 3, 2, 1, 0]].tolist()
v_val = [[valor for valor in lista if valor!='nulo'] for lista in v_val]
final = [' '.join(row).strip() for row in v_val]

#%%

pred_final = pd.DataFrame()
pred_final['id']=valid['id'].copy()
pred_final['productos'] = final
pred_final['id']=pred_final['id'].astype(np.int64)
pred_final=pred_final.fillna(" ")
pred_final.to_csv('Resultados/NaibeBayes.csv',index=False)
#%%

#Importamos librerias necesarias.
from keras.optimizers import SGD, Adam

## Ejecutare 3 redes neuronales identicas en capas y nodos. Invocando MiRed()
## Pero la optimizaremos usando el Grandiente Descendente Estocastico (SGD en ingles)
## Es a este operador al que uno le entrega la Tasa de Aprendizaje (Learning Rate)
## Otro ejemplo creando un optimizador usando Adam
adam = Adam(lr=0.01)

lr_to_test = [ .000001,  0.01, 0.1, 1, 10]  #tres tasas de aprendizaje diferentes, una pequeña, mediana y grande


## 1) Que cree que ocurriria si algun peso es mayor que 1??

for alfa in lr_to_test:
    print('\nProbando el modelo with learning rate: %f\n'%alfa )
    modelo = MiRed() ## creo una red limpia. 
    
    my_optimizer = SGD(lr = alfa) ##metodo de optimizacion
    
    modelo.compile(optimizer=my_optimizer,loss='mean_squared_error',metrics=['accuracy'])
    
    #results = modelo.fit(X_train, Y_train, epochs=1000, verbose=0)
    results = modelo.fit(X_train_us,y_train_us, epochs=100, verbose=0)
    
        
    print("Train-Accuracy: {:.2%}".format(np.mean(results.history["accuracy"][-11:-1])),"+/- {:.6%}".format(np.std(results.history["accuracy"][-11:-1])))
    print("Train-Loss: {:.2%}".format(np.mean(results.history["loss"][-11:-1])),"+/- {:.6%}".format(np.std(results.history["loss"][-11:-1])))




    

