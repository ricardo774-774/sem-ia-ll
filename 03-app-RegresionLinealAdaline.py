import numpy as np
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Adaline:
    def __init__(self, learning_rate=0.001, epochs=10000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        for _ in range(self.epochs):
            output = self.net_input(X)
            errors = (y - output)
            self.weights += self.learning_rate * X.T.dot(errors)
            self.bias += self.learning_rate * errors.sum()

    def net_input(self, X):
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        return self.net_input(X)

class RegressionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Regresión Lineal con ADALINE")
        self.points = []

        # Crear la figura y el eje de matplotlib
        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(-10, 10)  # Límites de -10 a 10 para mostrar los 4 cuadrantes completos
        self.ax.set_ylim(-10, 10)
        self.ax.set_title("Plano Cartesiano")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")

        # Agregar líneas para los ejes X e Y
        self.ax.axhline(0, color='black', linewidth=0.5)  # Eje X
        self.ax.axvline(0, color='black', linewidth=0.5)  # Eje Y

        # Incrustar la figura en Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()
        self.canvas.mpl_connect("button_press_event", self.add_point)

        # Botón de activación
        self.activate_button = tk.Button(root, text="Activar ADALINE", command=self.activate_adaline)
        self.activate_button.pack()

    def add_point(self, event):
        if event.inaxes != self.ax:
            return
        x, y = event.xdata, event.ydata
        self.points.append((x, y))
        self.ax.plot(x, y, 'bo')  # Punto azul
        self.canvas.draw()

    def activate_adaline(self):
        if len(self.points) < 2:
            print("Se necesitan al menos dos puntos para realizar la regresión lineal.")
            return
        
        X = np.array([[p[0]] for p in self.points])
        y = np.array([p[1] for p in self.points])
        
        adaline = Adaline(learning_rate=0.001, epochs=10000)
        adaline.fit(X, y)

        # Generar predicciones para un rango amplio de X
        X_plot = np.linspace(-10, 10, 100).reshape(-1, 1)
        y_pred = adaline.predict(X_plot)

        # Dibujar la línea de regresión
        self.ax.plot(X_plot, y_pred, 'r-', label='Línea de regresión')
        self.ax.legend()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = RegressionApp(root)
    root.mainloop()