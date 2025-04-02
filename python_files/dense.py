import tensorflow as tf
import math
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

#Funcion de normalizacion para los datos (Pasar valor de los pixeles de 0-255 a 0-1)
#Hace que la red aprenda mejor y mas rapido
def normalizar(imagenes, etiquetas):
  imagenes = tf.cast(imagenes, tf.float32)
  imagenes /= 255 #Aqui lo pasa de 0-255 a 0-1
  return imagenes, etiquetas


#Descargar set de datos de MNIST (Numeros escritos a mano, etiquetados)
datos, metadatos = tfds.load('mnist', as_supervised=True, with_info=True)

#Obtenemos en variables separadas los datos de entrenamiento (60k) y pruebas (10k)
datos_entrenamiento, datos_pruebas = datos['train'], datos['test']

#Normalizar los datos de entrenamiento y pruebas con la funcion que hicimos
datos_entrenamiento = datos_entrenamiento.map(normalizar)
datos_pruebas = datos_pruebas.map(normalizar)

#Agregar a cache (usar memoria en lugar de disco, entrenamiento mas rapido)
datos_entrenamiento = datos_entrenamiento.cache()
datos_pruebas = datos_pruebas.cache()

#Crear el modelo
dense_model = tf.keras.Sequential([
    tf.keras.layers.Flatten( input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax) #Para redes de clasificacion
])

#Compilar el modelo
dense_model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

#Los numeros de datos en entrenamiento y pruebas (60k y 10k)
num_ej_entrenamiento = metadatos.splits["train"].num_examples
num_ej_pruebas = metadatos.splits["test"].num_examples

#El trabajo por lotes permite que entrenamientos con gran cantidad de datos se haga de manera mas eficiente
TAMANO_LOTE = 32

#Shuffle y repeat hacen que los datos esten mezclados de manera aleatoria para que la red
#no se vaya a aprender el orden de las cosas
datos_entrenamiento = datos_entrenamiento.repeat().shuffle(num_ej_entrenamiento).batch(TAMANO_LOTE)
datos_pruebas = datos_pruebas.batch(TAMANO_LOTE)

dense_hist = dense_model.fit(datos_entrenamiento, epochs=20, validation_data=datos_pruebas, steps_per_epoch= math.ceil(num_ej_entrenamiento/TAMANO_LOTE), verbose=1 )

# Gráfica de Accuracy
plt.figure(figsize=(8, 6))
plt.plot(dense_hist.epoch, dense_hist.history['accuracy'], 'r',label='precisión')
plt.title(u'Categorical Crossentropy')
plt.xlabel(u'época')
plt.ylabel(r'$accuracy(\omega, b)$')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('dense_acc_graph.png', bbox_inches='tight', dpi=300)
plt.close()

# Gráfica de Loss
plt.figure(figsize=(8, 6))
plt.plot(dense_hist.epoch, dense_hist.history['loss'], 'b',label='error')
plt.title(u'Categorical Crossentropy')
plt.xlabel(u'época')
plt.ylabel(r'$loss(\omega, b)$')
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig('dense_loss_graph.png', bbox_inches='tight', dpi=300)
plt.close()

dense_model.save('dense_model.keras') 
dense_model.export('dense_model')


