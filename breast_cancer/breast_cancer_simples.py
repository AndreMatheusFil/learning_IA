import pandas as pd
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, accuracy_score

previsores = pd.read_csv("entradas_breast.csv")
classes = pd.read_csv("saidas_breast.csv")
previsores_treinamento,previsores_teste,classes_treinamento,classes_teste = train_test_split(previsores,classes,test_size=0.25)
classificador = Sequential()
classificador.add(Dense(units = 16,activation="relu", kernel_initializer="random_uniform",input_dim= 30))
classificador.add(Dense(units = 16,activation="relu", kernel_initializer="random_uniform"))
classificador.add(Dense(units = 1, activation='sigmoid'))
otimizadores = optimizers.Adam(lr = 0.01, decay = 0.1, clipvalue = 0.5)
classificador.compile(optimizer=otimizadores,loss="binary_crossentropy",metrics=["binary_accuracy"])
#classificador.compile(optimizer="adam",loss="binary_crossentropy",metrics=["binary_accuracy"])
classificador.fit(previsores_treinamento,classes_treinamento,batch_size=10,epochs=100)

peso0 = classificador.layers[0].get_weights()
print(peso0)
peso1 = classificador.layers[1].get_weights()
print(peso1)
peso2 = classificador.layers[2].get_weights()
print(peso2)

previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)
precisao = accuracy_score(classes_teste,previsoes)
matriz = confusion_matrix(classes_teste,previsoes)
resultado = classificador.evaluate(previsores_teste,classes_teste)
print(resultado)
