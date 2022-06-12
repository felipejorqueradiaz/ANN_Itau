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
import sklearn.metrics as metrics
from keras.callbacks import EarlyStopping

#%% Carga de dataset
path='C:/Users/Asus/Documents/GitHub/ANN_Itau'
#path = 'C:/Users/Felipe/Documents/Github/ANN_Itau'


os.chdir(path)

target= pd.read_pickle('Datos/final/Target.pkl', compression= 'zip')
#%%

product_list = ['A-A',
                'B-B',
                'C-D',
                'D-E',
                'E-E']

#%% Lectura de Train/Test

train = {}
test = {}
val = {}

for prod in product_list:
    train[prod] = pd.read_pickle('Datos/final/{}_train.pkl'.format(prod), compression= 'zip')
    test[prod] = pd.read_pickle('Datos/final/{}_test.pkl'.format(prod), compression= 'zip')
    val[prod] = pd.read_pickle('Datos/final/{}_val.pkl'.format(prod), compression= 'zip')


#%%



prod='A-A'
rus = RandomUnderSampler(random_state=0)
    
X = train[prod].drop(['id', 'Periodo', 'Target'], axis=1)
y = train[prod]['Target']
    
X_train_us, y_train_us = rus.fit_resample(X, y)
    
X_test = test[prod].drop(['id', 'Periodo', 'Target'], axis=1)
y_test = test[prod]['Target']
id_per = test[prod][['id', 'Periodo']]
 
#%%
########################################
from tensorflow import keras

n_modelos = 4

##### cada modelo tendra n_nodos_capa en cada capa oculta
n_nodos_capa = 10
n_features = X.shape[1]
n_classes = 1


##########FUNCION PARA CREAR MODELOS
## Una funcion para crear nuesras redes
## que es más general que la vista anteriormente.
def create_custom_model(input_dim, output_dim, nodes, n=5, name='model'):
    def create_model():
        # Create model
        model = Sequential(name=name)
        for i in range(n):
            model.add(Dense(nodes, input_dim=input_dim, activation='relu'))

        model.add(Dense(output_dim, activation='softmax'))
        opt = keras.optimizers.Adam(learning_rate=0.1)
        model.compile(loss='categorical_crossentropy', 
                      optimizer=opt, #antes 'adam'
                      metrics=['accuracy'])
    
        return model
    return create_model


########################################
print("======GENERALES=======")
print("n_features \t=\t"+str(n_features))
print("n_classes \t=\t"+str(n_classes))
print("n_modelos \t=\t"+str(n_modelos-1))
print("======================\n\n")

#%%
########################################
## AQUI GENERO 3 MODELOS ..
models = [create_custom_model(n_features, n_classes, n_nodos_capa, i, 'model_{}'.format(i)) for i in range(1, n_modelos+1)]

for create_model in models:
    create_model().summary()
    


#####################################################3
####################################################
#%%


history_dict = {} ##para los valores
matrices = {} ##para las matrices de confusion
# cb= TensorBoard(log_dir=PATH_LOG, histogram_freq=1)
cb =EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
real = pd.DataFrame()
pred = pd.DataFrame()
valid = pd.DataFrame()

for create_model in models:    
    
    model = create_model()
    
    print('Model name:', model.name)
    
    history_callback = model.fit(X_train_us, y_train_us,
                                 #batch_size=10,
                                 epochs=400,
                                 verbose=1,
                                 #validation_data=(X_test, Y_test),## esto es analogo a validation_split = 0.1
                                 validation_split = 0.1,
                                 callbacks=[cb])

    
    score = model.evaluate(X_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print(score)
    #var_dump(score)
    
    ### Genero la predicción con el modelo
    y_pred_proba = model.predict(X_test)     
    # y_pred[y_pred <= 0.5] = 0
    # y_pred[y_pred > 0.5] = 1
    pred[prod] = y_pred_proba
    real[prod] = y_test
    
print('-----------\nPRODUCTO {}\n'.format(prod),classification_report(y_test, y_pred_proba),'\n\n')
real = pd.concat([real, id_per], axis = 1, ignore_index=True)
pred = pd.concat([pred, id_per.reset_index(drop = True)], axis = 1, ignore_index=True)
valid = pd.concat([valid, id_per_val.reset_index(drop = True)], axis = 1, ignore_index=True)

real.columns = product_list + ['id', 'Periodo']
pred.columns = product_list + ['id', 'Periodo']
valid.columns = product_list + ['id']


    
    
# #VALIDACION
# X_val = val[prod].drop(['id', 'Periodo', 'Target'], axis=1)
#     id_per_val = val[prod][['id']]
#     valid[prod] = model.predict_proba(X_val).T[1]

# real = pd.concat([real, id_per], axis = 1, ignore_index=True)
# pred = pd.concat([pred, id_per.reset_index(drop = True)], axis = 1, ignore_index=True)
# valid = pd.concat([valid, id_per_val.reset_index(drop = True)], axis = 1, ignore_index=True)

# real.columns = product_list + ['id', 'Periodo']
# pred.columns = product_list + ['id', 'Periodo']
# valid.columns = product_list + ['id']