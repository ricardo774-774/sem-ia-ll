import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier

# Datos y etiquetas
X = []
y = []

# Red neuronal con una capa oculta de 10 neuronas
clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=0)

# Función que se ejecuta al hacer clic sobre el gráfico
def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        # Asignación de clase por botón
        if event.button == 1:
            label = 0
            color = 'blue'
        elif event.button == 3:
            label = 1
            color = 'red'
        else:
            return #Ignorar otros botones

        # Agregar punto y etiqueta
        X.append([event.xdata, event.ydata])
        y.append(label)
        plt.scatter(event.xdata, event.ydata, color=color, s=50)

        # Entrenar si hay suficientes puntos
        if len(X) >= 5:
            clf.fit(X, y)
            plot_decision_boundary()
        plt.draw()

# Función para graficar el plano de decisión
def plot_decision_boundary():
    x_min, x_max = -1.5, 1.5
    y_min, y_max = -1.5, 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    if len(X) >= 5:
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.coolwarm)

# Configurar gráfico
plt.figure(figsize=(8, 6))
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Clasificación de puntos no linealmente separables con red neuronal")
plt.grid(True)

# Conectar evento de clic con la función
plt.gcf().canvas.mpl_connect('button_press_event', onclick)

# Mostrar gráfico
plt.show()