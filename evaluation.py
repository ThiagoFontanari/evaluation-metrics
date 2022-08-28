from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
import keras
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Loading model's train and validation data
# Carregando os dados para treino e validação do modelo

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory='datasets/train/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(224,224,))
validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory='datasets/valid/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,  
    image_size=(224, 224,))

# Building a model using the transfer learning method, using imagenet weigths
# Construindo um modelo pelo método de transfer learning, usando os pesos da imagenet

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3), classes=2)
base_model.trainable = False
new_model = keras.Sequential()
new_model.add(base_model)
new_model.add(Flatten())
new_model.add(Dense(512, activation='relu'))
new_model.add(Dense(2, activation='softmax'))
new_model.summary()

# Compiling and training the model
# Compilando e treinando o modelo

new_model.compile(optimizer=Adam(lr=0.001),loss='CategoricalCrossentropy',metrics=['accuracy'])
history = new_model.fit(train_ds, epochs=20, validation_data=validation_ds)

# Generating a graph to demonstrate the model's evolution during training
# Gerando um gráfico para demonstrar a evolução do modelo durante o treinamento
metrics = history.history
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()

# Generating predictions and evaluations with the model from data not used in training
# Gerando previsões e avaliações com o modelo a partir de dados não utilizados no treinamento

# Loading new data for tests and evaluation
# Carregando novos dados para realização dos testes e avaliações

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory='datasets/test/',
    labels='inferred',
    label_mode='categorical',
    image_size=(224, 224,),
    shuffle=False)

# Storing test data labels
# Armazenando as labels dos dados de teste
labels = np.resize(np.array([label[:,1] for image, label in test_ds]), (22))

# Making predictions on the test data
# Realizando previsões sobre os dados de teste
predictions = np.array((new_model.predict(test_ds)).argmax(axis=-1))

# Calculating and printing test accuracy
# Calculando e imprimindo a acurácia do teste
test_acc = sum(predictions == labels) / len(labels)
print(f'Test set accuracy: {test_acc:.0%}')

# Generating the confusion matrix from the predictions made
# Gerando a matriz de confusão a partir das previsões realizadas
cfm = tf.math.confusion_matrix(labels,predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cfm, annot=True, fmt='g', 
            xticklabels=('Defective','Non Defective'), yticklabels=('Defective',"Non Defective"))
plt.xlabel('Prediction')
plt.ylabel('Real')
plt.show()

