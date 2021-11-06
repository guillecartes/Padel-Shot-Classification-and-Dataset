# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 08:45:58 2021

@author: Guillermo
"""

# cnn model
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import collections

directorio_dataset = '/Users/guill/OneDrive/Escritorio/Master/TFM/base de datos/guardados/Dataset12.csv'

ventana=40
clases=13
clases_str = ['D','R','DP','RP','GD','GR','GDP','GRP','VD','VR','B','RM','S']

import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Prediction')
    plt.xticks(range(13), clases_str)
    plt.yticks(range(13), clases_str)


# load the dataset, returns train and test X and y elements
def load_dataset_group(prefix='',normalize=False):
    
    datos=pd.read_csv(directorio_dataset)
     
    #feature engineering, realizamos un nuevo data frame con
    # las siguientes características para cada golpe

    """
    Ax_mean: valor medio aceleración eje x
    Ay_mean: valor medio aceleración eje y
    Az_mean: valor medio aceleración eje z
    
    Vx_mean: valor medio velocidad eje x
    Vy_mean: valor medio velocidad eje y
    Vz_mean: valor medio velocidad eje z
    
    Ax_max: valor máximo aceleración eje x
    Ay_max: valor máximo aceleración eje y
    Az_max: valor máximo aceleración eje z
    
    Vx_max: valor máximo velocidad eje x
    Vy_max: valor máximo velocidad eje y
    Vz_max: valor máximo velocidad eje z
    
    Ax_min: valor mínimo aceleración eje x
    Ay_min: valor mínimo aceleración eje y
    Az_min: valor mínimo aceleración eje z
    
    Vx_min: valor mínimo velocidad eje x
    Vy_min: valor mínimo velocidad eje y
    Vz_min: valor mínimo velocidad eje z
    
    """
    
    datos_features = pd.DataFrame()
    
    datos_features["Ax_mean"] = datos.loc[:, "Ax0":"Ax39"].mean(axis=1)
    datos_features["Ay_mean"] = datos.loc[:, "Ay0":"Ay39"].mean(axis=1)
    datos_features["Az_mean"] = datos.loc[:, "Az0":"Az39"].mean(axis=1)
    
    datos_features["Vx_mean"] = datos.loc[:, "Vx0":"Vx39"].mean(axis=1)
    datos_features["Vy_mean"] = datos.loc[:, "Vy0":"Vy39"].mean(axis=1)
    datos_features["Vz_mean"] = datos.loc[:, "Vz0":"Vz39"].mean(axis=1)
    
    datos_features["Ax_max"] = datos.loc[:, "Ax0":"Ax39"].max(axis=1)
    datos_features["Ay_max"] = datos.loc[:, "Ay0":"Ay39"].max(axis=1)
    datos_features["Az_max"] = datos.loc[:, "Az0":"Az39"].max(axis=1)
    
    datos_features["Vx_max"] = datos.loc[:, "Vx0":"Vx39"].max(axis=1)
    datos_features["Vy_max"] = datos.loc[:, "Vy0":"Vy39"].max(axis=1)
    datos_features["Vz_max"] = datos.loc[:, "Vz0":"Vz39"].max(axis=1)
    
    datos_features["Ax_min"] = datos.loc[:, "Ax0":"Ax39"].min(axis=1)
    datos_features["Ay_min"] = datos.loc[:, "Ay0":"Ay39"].min(axis=1)
    datos_features["Az_min"] = datos.loc[:, "Az0":"Az39"].min(axis=1)
    
    datos_features["Vx_min"] = datos.loc[:, "Vx0":"Vx39"].min(axis=1)
    datos_features["Vy_min"] = datos.loc[:, "Vy0":"Vy39"].min(axis=1)
    datos_features["Vz_min"] = datos.loc[:, "Vz0":"Vz39"].min(axis=1)
    
    datos_features["id"] = datos["id"].astype(int)
    
    datos_features["tipo_golpe"] = datos["tipo_golpe"].astype(int)
        
    y = datos_features.loc[:, "tipo_golpe"].to_numpy()
    # Grafica con la cantidad de golpes que hay para cada tipo
    #print(collections.Counter(y))
    #plt.hist(y,bins=clases)
    #plt.plot()
    
    # Se divide el Dataset en Train y Test
    # Se barajan antes de dividirlo (shuffle=True)
    # Se dividen de forma que entrenamiento y test estén balanceados (stratify=y)
    # Se dividen de forma aleatoria, pero siempre la misma (random_state=int)
    trainingSet, testSet = train_test_split(datos_features, test_size=0.2,shuffle=True,stratify=y,random_state=5)
    #trainingSet, testSet = train_test_split(datos, test_size=0.2) 
    
    #Se recoge los datos de Train y Set:
    name_ax = ["Ax_mean","Ax_max","Ax_min"]
    name_ay = ["Ay_mean","Ay_max","Ay_min"]
    name_az = ["Az_mean","Az_max","Az_min"]
    name_vx = ["Vx_mean","Vx_max","Vx_min"]
    name_vy = ["Vy_mean","Vy_max","Vy_min"]
    name_vz = ["Vz_mean","Vz_max","Vz_min"]
        
    trainX = trainingSet[name_ax+name_ay+name_az+name_vx+name_vy+name_vz]
    trainy = trainingSet[['tipo_golpe']]
    testX = testSet[name_ax+name_ay+name_az+name_vx+name_vy+name_vz]
    testy = testSet[['tipo_golpe']]
    
    trainX=trainX.to_numpy()
    trainy=trainy.to_numpy()
    testX=testX.to_numpy()
    testy=testy.to_numpy()
        
    #guardo todos los datos de cada GDL por separado
    n_fe = 3
    datos_trainX=trainX.shape[0]
    trainX_accel_x = [trainX[i][0:n_fe] for i in range(datos_trainX)]
    trainX_accel_y = [trainX[i][n_fe:n_fe*2] for i in range(datos_trainX)]
    trainX_accel_z = [trainX[i][n_fe*2:n_fe*3] for i in range(datos_trainX)]
    trainX_gyros_x = [trainX[i][n_fe*3:n_fe*4] for i in range(datos_trainX)]
    trainX_gyros_y = [trainX[i][n_fe*4:n_fe*5] for i in range(datos_trainX)]
    trainX_gyros_z = [trainX[i][n_fe*5:n_fe*6] for i in range(datos_trainX)]
    
    datos_testX=testX.shape[0]
    testX_accel_x = [testX[i][0:n_fe] for i in range(datos_testX)]
    testX_accel_y = [testX[i][n_fe:n_fe*2] for i in range(datos_testX)]
    testX_accel_z = [testX[i][n_fe*2:n_fe*3] for i in range(datos_testX)]
    testX_gyros_x = [testX[i][n_fe*3:n_fe*4] for i in range(datos_testX)]
    testX_gyros_y = [testX[i][n_fe*4:n_fe*5] for i in range(datos_testX)]
    testX_gyros_z = [testX[i][n_fe*5:n_fe*6] for i in range(datos_testX)]
    
    
    #se crea trainX con dimension (datos_trainX,n_fe,GDL)
    trainX = np.array([trainX_accel_x,trainX_accel_y,trainX_accel_z,trainX_gyros_x,trainX_gyros_y,trainX_gyros_z])
    
    testX = np.array([testX_accel_x,testX_accel_y,testX_accel_z,testX_gyros_x,testX_gyros_y,testX_gyros_z])
       
    #Necesito una matriz de (datos_trainX, n_fe, GDL), pero tengo en trainX (GDL, datos_trainX, n_fe)
    trainX_ordenada = np.ones((trainX.shape[1],n_fe,6))
    for i in range(trainX.shape[1]):
        for j in range(trainX.shape[2]):
            for k in range(trainX.shape[0]):
                trainX_ordenada[i][j][k] = trainX[k][i][j]
                
    #Necesito una matriz de (datos_testX, ventana, GDL), pero tengo en testX (GDL, datos_testX, ventana)
    testX_ordenada = np.ones((testX.shape[1],n_fe,6))
    for i in range(testX.shape[1]):
        for j in range(testX.shape[2]):
            for k in range(testX.shape[0]):
                testX_ordenada[i][j][k] = testX[k][i][j]

    #Guardo la característica que me interesa del golpe para estudiar los fallos
    test_deportistas = testSet.loc[:, "id"].to_numpy()
    
    return trainX_ordenada, trainy, testX_ordenada, testy, test_deportistas

# load the dataset, returns train and test X and y elements
def load_dataset(directorio_dataset):
	
    # carga train y test
	trainX, trainy, testX, testy, test_deportistas = load_dataset_group(directorio_dataset)
	print(trainX.shape, trainy.shape)
	print(testX.shape, testy.shape)
    
	# one hot encode y
	trainy = to_categorical(trainy)
	testy = to_categorical(testy)
	print(trainX.shape, trainy.shape, testX.shape, testy.shape)
	return trainX, trainy, testX, testy, test_deportistas


# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy, test_deportistas, n_filters, epochs, batch_size):
	verbose = 0
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    
	model = Sequential()
  
	#Flatten convierte las características en un vector para pasarselo a la capa densa.
	model.add(Flatten())
	model.add(Dense(n_filters, activation='relu'))
	#model.add(Dense(n_filters/2, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	train_log = model.fit(trainX, trainy, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=verbose)
	
	#model.summary()
    
	# evaluate model
	loss, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    

	#model.save_weights("CNN.h5")

    # MATRIZ DE CONFUSION:
    # Predecimos las clases para los datos de test
	Y_pred = model.predict(testX)
    # Convertimos las predicciones en one hot encoding
	Y_pred_classes = np.argmax(Y_pred, axis = 1) 
    # Convertimos los datos de test en one hot encoding
	Y_true = np.argmax(testy, axis = 1) 
    # Computamos la matriz de confusion
	confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
	#print(confusion_mtx)
    # Mostramos los resultados
	#plt.figure()
	#plot_confusion_matrix(cm = confusion_mtx, classes = range(13)) 
     
	fallos=identify_faults(test_deportistas,Y_true,Y_pred_classes) 
	#print("Los fallos corresponden a: ",collections.Counter(fallos))

	# grafica con la función de coste
	loss = train_log.history['loss']
	val_loss = train_log.history['val_loss']
	epochs = range(1, len(loss) + 1)
	plt.figure()
	plt.plot(epochs, loss, 'b', label='Training loss')
	plt.plot(epochs, val_loss, 'r', label='Validation loss')
	plt.title('Training and validation loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.show()
    
	return accuracy

def identify_faults(test_deportistas, dataY, predictions):
    deportista=[]
    for i in range(len(dataY)):
        if dataY[i]!=predictions[i]:
            deportista.append(test_deportistas[i])
            #print("Error: Realidad:",dataY[i],"Prediccion:",predictions[i])
        
    return deportista

# summarize scores
def summarize_results(scores, filters, epochs, batch_size):
	# Esta funcion saca una grafica con un diagrama de bigotes de los resultados
	# Saca la gráfica para los distintos numero de filtros provados para una misma configuracion, por lo que epoch y batch_size serán fijos en la gráfica
	# Si se está realizando la busqueda en rejilla con distintos valores para epoch y batch, sacará una gráfica distinta para cada combinación
	# summarize mean and standard deviation
   
	best_accuracy = 0
	best_params = []
	for i in range(len(scores)):
		m, s = mean(scores[i]), std(scores[i])
		print('Epoch=%d; Batch_size=%d; Filtros=%d: %.3f%% (+/-%.3f)' % (epochs, batch_size, filters[i], m, s))
		if m>best_accuracy:
			best_accuracy = m
			best_params = [m, s, epochs, batch_size, filters[i]]
	# boxplot of scores
	#pyplot.figure( figsize=(10,7))
	pyplot.figure()
	pyplot.boxplot(scores, labels=filters)
	pyplot.title('Accuracy para Epoch=%d, Batch_size=%d, Filtros capa 1=%d' % (epochs, batch_size, filters[i]))
	#pyplot.title('Accuracy para Epoch=%d, Batch_size=%d, Filtros capa 1=%d, Filtros capa 2=%d' % (epochs, batch_size, filters[i], filters[i]/2))
	pyplot.xlabel("Número de filtros")
	pyplot.ylabel("Accuracy (%)")
	pyplot.grid(linestyle='-', linewidth=0.3)
	#pyplot.savefig('exp_cnn_filters.png')
    
	return best_params


# run an experiment
def run_experiment(filters, epochs, batch_size, repeats=10):
	# load data
	trainX, trainy, testX, testy, test_deportistas = load_dataset(directorio_dataset)
	# test each parameter
	best_accuracy = 0
	best_params = []
	for i in epochs:
		for j in batch_size:      
			all_scores = list()
			for p in filters:
				# repeat experiment
				scores = list()
				for r in range(repeats):
					score = evaluate_model(trainX, trainy, testX, testy, test_deportistas, p, i, j)
					score = score * 100.0
					print('>p=%d #%d: %.3f' % (p, r+1, score))
					scores.append(score)
				all_scores.append(scores)
			# summarize results
			params = summarize_results(all_scores, filters, i, j)
			if params[0]>best_accuracy:
				best_accuracy = params[0]
				best_params = params
                

	print("Best params:")
	print('Epoch=%d; Batch_size=%d; Filtros=%d: %.3f%% (+/-%.3f)' % (best_params[2], best_params[3], best_params[4], best_params[0], best_params[1],))
	 
# run the experiment
epochs = [100]
batch_size = [50]
filters = [1500]

# =============================================================================
# epochs = [40, 70, 100, 200]
# batch_size = [30, 50, 70]
# filters = [500, 1000, 1500, 2000]
# =============================================================================

run_experiment(filters, epochs, batch_size)





