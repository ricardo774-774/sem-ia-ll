import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Paso 1: Cargar los archivos
df_entradas = pd.read_csv('act4-recursos/x.csv')
df_deseado = pd.read_csv('act4-recursos/deseado.csv')

# Convertir los datos de entrada y salida a matrices
X = df_entradas.values
D = df_deseado.values

# Normalización de las entradas
scaler = StandardScaler()
X = scaler.fit_transform(X)  # Normalización entre -1 y 1

# Parámetros de la red neuronal
np.random.seed(42)
pesos = np.random.randn(2, 3) * np.sqrt(1 / 2)  # Inicialización de pesos con Xavier
sesgo = np.random.uniform(-1, 1, (1, 3))  # Sesgo para cada neurona
learning_rate = 0.5  # Tasa de aprendizaje
epochs = 10000  # Número de épocas

# Función de activación: Sigmoide
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la sigmoide
def sigmoide_derivada(x):
    return x * (1 - x)

# Propagación hacia adelante
def forward_propagation(X, pesos, sesgo):
    z = np.dot(X, pesos) + sesgo  # Producto punto entre entradas y pesos + sesgo
    return sigmoide(z)  # Aplicar activación

# Entrenamiento de la red neuronal
def entrenar_red_neuronal(X, D, pesos, sesgo, learning_rate, epochs):
    errores = []
    for epoch in range(epochs):
        salida = forward_propagation(X, pesos, sesgo)
        error = D - salida
        ajuste = error * sigmoide_derivada(salida)

        # Ajustar los pesos y el sesgo
        pesos += learning_rate * np.dot(X.T, ajuste)
        sesgo += learning_rate * np.sum(ajuste, axis=0, keepdims=True)

        # Guardar error promedio
        errores.append(np.mean(np.abs(error)))

    return pesos, sesgo, errores

# Entrenar la red neuronal
pesos, sesgo, errores = entrenar_red_neuronal(X, D, pesos, sesgo, learning_rate, epochs)

# Paso 2: Obtener la salida final después del entrenamiento
salida_entrenada = forward_propagation(X, pesos, sesgo)

# Redondear las predicciones a 0 o 1
salida_entrenada_binaria = np.round(salida_entrenada)

# Paso 3: Visualización de las salidas deseadas vs predicciones
desplazamiento = 0.05
fig1, ax1 = plt.subplots()  # Usamos fig1 y ax1 explícitamente para la figura 1

# Graficar valores deseados y predicciones
elementos = [('red', 'Deseado d1', 'x', 'Predicción d1'),
             ('green', 'Deseado d2', 'D', 'Predicción d2'),
             ('blue', 'Deseado d3', 'x', 'Predicción d3')]

for i, (color, label_d, marker_p, label_p) in enumerate(elementos):
    ax1.scatter(range(len(D)), D[:, i], color=color, label=label_d, alpha=0.7)
    ax1.scatter(np.array(range(len(D))) + desplazamiento, salida_entrenada[:, i],
                color=color, marker=marker_p, label=label_p, alpha=0.4)

ax1.set_ylim(-0.1, 1.1)
ax1.set_xlabel('Ejemplos')
ax1.set_ylabel('Activación (0 o 1)')
ax1.legend()
plt.title('Visualización de Salidas Deseadas vs Predicciones')

# Mostrar figura 1 sin bloquear el flujo
plt.show()

# Paso 4: Visualización de las matrices de confusión
fig3, axs3 = plt.subplots(1, 3, figsize=(15, 5))  # Usamos fig3 y axs3 explícitamente para la figura 3

for i in range(3):
    matriz_confusion = confusion_matrix(D[:, i], salida_entrenada_binaria[:, i])
    sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues', ax=axs3[i])
    axs3[i].set_title(f'Neurona d{i+1}')
    axs3[i].set_xlabel('Predicciones')
    axs3[i].set_ylabel('Verdaderos')

plt.tight_layout()

# Mostrar figura 3 sin bloquear el flujo
plt.show()

# Gráfico del error durante el entrenamiento
plt.plot(range(epochs), errores)
plt.xlabel("Épocas")
plt.ylabel("Error Promedio")
plt.title("Evolución del Error en el Entrenamiento")

# Mostrar el gráfico de error
plt.show()
