import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


base = pd.read_csv('iris.csv')
previsores= base.iloc[:,0:4].values
classes = base.iloc[:,4].values
labelencoder = LabelEncoder()
classes =labelencoder.fit_transform(classes)
classe_dummy = np_utils.to_categorical(classes)
previsores_treinamento,previsores_teste,classes_treinamento,classes_teste = train_test_split(previsores,classe_dummy,test_size=0.25)
print(len(previsores_treinamento))
classificador = Sequential()
classificador.add(Dense(units=4,activation="relu",input_dim=4))
classificador.add(Dense(units=4,activation='relu'))
classificador.add(Dense(units=3,activation="softmax"))
classificador.compile(optimizer="adam",loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])

classificador.fit(previsores_treinamento, classes_treinamento, batch_size=10,epochs=1000)
resultado = classificador.evaluate(previsores_teste,classes_teste)
print(resultado)