# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 08:45:58 2021

@author: Guillermo
"""

import pandas as pd

datos = pd.read_csv("/Users/guill/OneDrive/Escritorio/Master/TFM/base de datos/guardados/Dataset12.csv")

#print(datos.shape)
#print(datos.info())

#%% eliminamos las columnas que no nos interesan

datos.drop(columns = ["mano", "reves", "altura", "edad", "sexo", "nivel","id", "numero_golpe", "tiempo_golpe"], inplace=True)


#%% nuevos datos que tenemos

print(datos.info())
print(datos.shape)

#%% dividimos los datos

from sklearn.model_selection import train_test_split

X = datos.drop(columns = ["tipo_golpe"])
y = datos["tipo_golpe"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=5)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

#%% matriz de confusión 
    
import numpy as np
import matplotlib.pyplot as plt
import itertools
    
from sklearn.metrics import confusion_matrix
    
golpes = ['D','R','DP','RP','GD','GR','GDP','GRP','VD','VR','B','RM','S']
    
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
    plt.xticks(range(13), golpes)
    plt.yticks(range(13), golpes)

#%% entrenamiento modelo

from sklearn.metrics import accuracy_score
from sklearn import svm

def evaluate_model(param_C,kernel_type):    
    
    
    model = svm.SVC(C=param_C, decision_function_shape='ovr', kernel=kernel_type)
    
    model.fit(X_train, y_train)
    
    #%% resultados test
    
    ypred = model.predict(X_test)
       
    accuracy = accuracy_score(y_test, ypred)
    #print(accuracy) 
    
    cm = confusion_matrix(y_test, ypred)       
    
    #plt.figure()
    #plot_confusion_matrix(cm, classes = range(3))  

    return accuracy


from numpy import mean
from numpy import std

import matplotlib.pyplot as plt

# summarize scores
def summarize_results(scores, C, kernel):

	print(scores)
	best_accuracy = 0
	best_params = []	
	for i in range(len(scores)):
		m, s = mean(scores[i]), std(scores[i])
		print('Kernel=%s: %.3f%% (+/-%.3f)' % (kernel[i], m, s))
        
		score = scores[i]
		for j in range(len(C)):
			if score[j]>best_accuracy:
					best_accuracy = score[j]
					best_params = [score[j], C[j], kernel[i]]
            
	# boxplot of scores
	plt.figure()
	plt.boxplot(scores, labels=['linear', 'poly', 'rbf', 'sigmoid'])
	plt.title('Accuracy en función de Kernel y C')
	plt.xlabel("Kernel_type")
	plt.ylabel("Accuracy (%)")
	plt.grid(linestyle='-', linewidth=0.3)
        
	print('Best Params: Kernel=%s, C=%.2f: %.2f%%' % (best_params[2], best_params[1], best_params[0]))

	#Matriz de Confusion de mejores parametros
	model = svm.SVC(C=best_params[1], decision_function_shape='ovr', kernel=best_params[2])
	model.fit(X_train, y_train)
    
	ypred = model.predict(X_test)
       
	accuracy = accuracy_score(y_test, ypred)
	print(accuracy) 
    
	cm = confusion_matrix(y_test, ypred)       
    
	plt.figure()
	plot_confusion_matrix(cm, classes = range(3))

# run an experiment
def run_experiment(C, Kernel):

	all_scores = list()
	for i in Kernel:
		scores = list()
		for j in C:
			score = evaluate_model(j, i)
			score = score * 100.00
			print('>#%s #%.2f: %.2f' % (i, j, score))
			scores.append(score)
		all_scores.append(scores)
	# summarize results
	params = summarize_results(all_scores, C, Kernel)


# run the experiment

C = [0.01, 0.1, 0.5, 1, 2, 10, 12, 20, 100]
kernel = ['linear', 'poly', 'rbf', 'sigmoid']


run_experiment(C, kernel)