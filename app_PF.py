#-------------------------Módulos-------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')
nltk.download('stopwords')


#-------------------------Cargar datos del archivo y balancear muestras-------------------------
def cargar_y_balancear_datos(ruta_archivo):
    datos = pd.read_csv(ruta_archivo)
    no_spam = datos[datos['label'] == 'ham']
    spam = datos[datos['label'] == 'spam']
    no_spam_balanceado = no_spam.sample(n=len(spam), random_state=42)
    balanceado = pd.concat([no_spam_balanceado, spam]).reset_index(drop=True)
    balanceado['text'] = balanceado['text'].str.replace('Subject', '', regex=False)
    return balanceado


#-------------------------Limpieza de texto-------------------------
def eliminar_puntuaciones(texto):
    return texto.translate(str.maketrans('', '', string.punctuation))

def eliminar_stopwords(texto):
    palabras_stop = set(stopwords.words('english'))
    palabras = texto.lower().split()
    palabras_importantes = [palabra for palabra in palabras if palabra not in palabras_stop]
    return " ".join(palabras_importantes)

def limpiar_texto(datos):
    datos['text'] = datos['text'].apply(eliminar_puntuaciones)
    datos['text'] = datos['text'].apply(eliminar_stopwords)
    return datos


#-------------------------Visualización-------------------------
def graficar_distribucion(datos):
    sns.countplot(x='label', data=datos)
    plt.title("Distribución de correos electrónicos")
    plt.xticks(ticks=[0, 1], labels=['No Spam', 'Spam'])
    plt.show()

def graficar_nube_palabras(datos, etiqueta):
    texto = " ".join(datos[datos['label'] == etiqueta]['text'])
    wc = WordCloud(background_color='black', max_words=100, width=800, height=400).generate(texto)
    plt.figure(figsize=(7, 7))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Nube de palabras - Correos {etiqueta.title()}")
    plt.show()


#-------------------------Tokenización y secuencias-------------------------
def preprocesar_textos(textos_entrenamiento, textos_prueba, max_len=100):
    tokenizador = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizador.fit_on_texts(textos_entrenamiento)
    secuencias_entrenamiento = tokenizador.texts_to_sequences(textos_entrenamiento)
    secuencias_prueba = tokenizador.texts_to_sequences(textos_prueba)
    entrenamiento_relleno = pad_sequences(secuencias_entrenamiento, maxlen=max_len, padding='post', truncating='post')
    prueba_relleno = pad_sequences(secuencias_prueba, maxlen=max_len, padding='post', truncating='post')
    return tokenizador, entrenamiento_relleno, prueba_relleno


#-------------------------Definición del modelo-------------------------
def construir_modelo(tamaño_vocabulario, max_len=100):
    modelo = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim=tamaño_vocabulario, output_dim=64, input_length=max_len),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    modelo.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        optimizer='adam',
        metrics=['accuracy']
    )
    return modelo


#-------------------------Entrenamiento-------------------------
def entrenar_modelo(modelo, X_entrenamiento, Y_entrenamiento, X_prueba, Y_prueba):
    early_stop = EarlyStopping(patience=3, monitor='val_accuracy', restore_best_weights=True)
    reduccion_lr = ReduceLROnPlateau(patience=2, monitor='val_loss', factor=0.5)
    
    historial = modelo.fit(
        X_entrenamiento, Y_entrenamiento,
        validation_data=(X_prueba, Y_prueba),
        epochs=20,
        batch_size=32,
        callbacks=[early_stop, reduccion_lr]
    )
    return historial

def graficar_precision(historial):
    plt.plot(historial.history['accuracy'], label='Precisión Entrenamiento')
    plt.plot(historial.history['val_accuracy'], label='Precisión Validación')
    plt.title('Precisión del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    plt.show()

def graficar_perdida(historial):
    plt.plot(historial.history['loss'], label='Pérdida Entrenamiento')
    plt.plot(historial.history['val_loss'], label='Pérdida Validación')
    plt.title('Pérdida del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.show()


#-------------------------Evaluación-------------------------
def evaluar_modelo(modelo, X_prueba, Y_prueba):
    probabilidades = modelo.predict(X_prueba)
    predicciones = (probabilidades > 0.5).astype(int)
    print(classification_report(Y_prueba, predicciones))
    sns.heatmap(confusion_matrix(Y_prueba, predicciones), annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicciones")
    plt.ylabel("Reales")
    plt.title("Matriz de Confusión")
    plt.show()


#-------------------------Ejecución-------------------------
# Carga y limpieza de datos
datos = cargar_y_balancear_datos('emails_practica.csv')
datos = limpiar_texto(datos)
graficar_distribucion(datos)
graficar_nube_palabras(datos, 'ham')
graficar_nube_palabras(datos, 'spam')

# Preparar datos
textos_entrenamiento, textos_prueba, etiquetas_entrenamiento, etiquetas_prueba = train_test_split(
    datos['text'], datos['label'], test_size=0.2, random_state=42)

etiquetas_entrenamiento = (etiquetas_entrenamiento == 'spam').astype(int)
etiquetas_prueba = (etiquetas_prueba == 'spam').astype(int)

# Tokenización
tokenizador, secuencia_entrenamiento, secuencia_prueba = preprocesar_textos(textos_entrenamiento, textos_prueba)
tamaño_vocabulario = len(tokenizador.word_index) + 1

# Construcción y entrenamiento del modelo
modelo = construir_modelo(tamaño_vocabulario)
historial = entrenar_modelo(modelo, secuencia_entrenamiento, etiquetas_entrenamiento, secuencia_prueba, etiquetas_prueba)

# Evaluación
graficar_precision(historial)
graficar_perdida(historial)
evaluar_modelo(modelo, secuencia_prueba, etiquetas_prueba)
