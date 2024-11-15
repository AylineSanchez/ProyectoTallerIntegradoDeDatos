import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, Scrollbar, Canvas, Frame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Configuración de estilo para gráficos
sns.set(style="whitegrid")

# Definir rutas a los archivos
grades_file_path = r"C:\Users\aylin\OneDrive\Escritorio\Taller integrado de datos\Taller integrado de datos\Data\StudentGrades.txt"
root_dir = r"C:\Users\aylin\OneDrive\Escritorio\Taller integrado de datos\Taller integrado de datos\Data\Data"

# Leer las calificaciones y organizar los datos
raw_grades_data = pd.read_csv(grades_file_path, sep=' - ', engine='python', header=None, names=['Student', 'Grade'])
# Filtrar las filas para obtener solo las calificaciones
grades_data = raw_grades_data[raw_grades_data['Student'].str.contains(r'^S\d{2}$', na=False)].copy()
grades_data['Grade'] = pd.to_numeric(grades_data['Grade'], errors='coerce')

# Diccionario para almacenar DataFrames
dfs = {}

# Iterar sobre las carpetas de los participantes S1 a S10
for participant_folder in range(1, 11):  # Desde S1 a S10
    participant_folder_name = f"S{participant_folder}"
    participant_path = os.path.join(root_dir, participant_folder_name)
    
    # Iterar sobre las subcarpetas Final, Midterm 1, Midterm 2
    for subfolder in ["Final", "Midterm 1", "Midterm 2"]:
        subfolder_path = os.path.join(participant_path, subfolder)
        
        # Lista de nombres de archivos
        filenames = ["EDA.csv", "HR.csv", "TEMP.csv", "BVP.csv", "ACC.csv", "IBI.csv"]
        
        # Iterar sobre la lista de nombres de archivos
        for filename in filenames:
            file_path = os.path.join(subfolder_path, filename)
            
            # Comprobar si el archivo existe
            if os.path.isfile(file_path):
                try:
                    # Leer el archivo CSV en un DataFrame
                    if filename in ["EDA.csv", "HR.csv", "TEMP.csv", "BVP.csv", "ACC.csv"]:
                        # Leer la hora inicial y la frecuencia de muestreo
                        with open(file_path, 'r', encoding='latin-1') as f:
                            initial_time = f.readline().strip()
                            sample_rate = f.readline().strip()
                        # Leer los datos omitiendo las dos primeras filas
                        df = pd.read_csv(file_path, skiprows=2, encoding='latin-1')
                        df.attrs['initial_time'] = initial_time
                        df.attrs['sample_rate'] = sample_rate
                    else:
                        df = pd.read_csv(file_path, encoding='latin-1')
                    # Almacenar DataFrame en un diccionario con la clave compuesta por participante y subcarpeta
                    key = f"{participant_folder_name}_{subfolder}_{filename.split('.')[0]}"
                    dfs[key] = df
                except Exception as e:
                    print(f"Error al leer el archivo {file_path}: {e}")
            else:
                print(f"El archivo no existe: {file_path}")

# Interfaz de Visualización con Tkinter

class DataAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Análisis Exploratorio de Datos y Modelado Predictivo")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        self.create_widgets()
        self.current_canvas = None
        self.current_treeview = None

    def create_widgets(self):
        # Crear un marco para los botones y otro para el área de gráficos
        button_frame = Frame(self.root, bg='#f0f0f0')
        button_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        graph_frame = Frame(self.root, bg='#ffffff')
        graph_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        self.graph_frame = graph_frame

        # Crear botones para cada tipo de gráfico con estilo atractivo
        button_options = [
            ("Tendencias por Participante", self.plot_tendencias),
            ("Gráficos de Dispersión entre Señales Fisiológicas y Calificaciones", self.select_plot_dispersion),
            ("Datos Contextuales - Rendimiento Académico", self.plot_rendimiento_academico),
            ("Modelado Predictivo", self.modelado_predictivo),
            ("Técnicas Estadísticas para Resumir Hallazgos", self.tecnicas_estadisticas),
            ("Desviación Estándar Comparativa", self.plot_desviacion_estandar),
            ("Salir", self.root.quit)
        ]

        for text, command in button_options:
            btn = ttk.Button(button_frame, text=text, command=command)
            btn.pack(pady=5, fill=tk.X)
            btn.configure(style='TButton')

    def clear_canvas(self):
        if self.current_canvas:
            self.current_canvas.get_tk_widget().pack_forget()
            self.current_canvas = None
        if self.current_treeview:
            self.current_treeview.pack_forget()
            self.current_treeview = None

    def plot_tendencias(self):
        self.clear_canvas()
        participante = simpledialog.askstring("Participante", "Seleccione un participante (S1 - S10):")
        examen = simpledialog.askstring("Examen", "Seleccione el tipo de examen (Final, Midterm 1, Midterm 2, Todos):")
        variable = simpledialog.askstring("Variable Fisiológica", "Seleccione la variable fisiológica (TEMP, EDA, HR, BVP, ACC, IBI):")

        fig, ax = plt.subplots(figsize=(12, 8))

        if examen == "Todos":
            for subfolder in ["Final", "Midterm 1", "Midterm 2"]:
                key = f"{participante}_{subfolder}_{variable}"
                if key in dfs:
                    df = dfs[key]
                    ax.plot(df.index, df.iloc[:, 0], label=subfolder)
            ax.set_title(f'Tendencia de {variable} para el participante {participante} en todos los exámenes')
        else:
            key = f"{participante}_{examen}_{variable}"
            if key in dfs:
                df = dfs[key]
                ax.plot(df.index, df.iloc[:, 0], label=examen)
                ax.set_title(f'Tendencia de {variable} para el participante {participante} durante el {examen}')

        ax.set_xlabel('Tiempo')
        ax.set_ylabel(variable)
        ax.legend()
        self.current_canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        self.current_canvas.draw()
        self.current_canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)
        plt.close(fig)

    def select_plot_dispersion(self):
        variable = simpledialog.askstring("Variable Fisiológica", "Seleccione la variable fisiológica para graficar (TEMP, EDA, HR):")
        if variable in ["TEMP", "EDA", "HR"]:
            self.plot_dispersion(variable)

    def plot_dispersion(self, variable):
        self.clear_canvas()
        fig, ax = plt.subplots(figsize=(12, 8))
        avg_values = []
        for key, df in dfs.items():
            if variable in key:
                avg_values.append(df.iloc[:, 0].mean())
        if len(avg_values) > 0:
            calificaciones = grades_data['Grade'].values[:len(avg_values)]
            sns.scatterplot(x=avg_values, y=calificaciones, ax=ax, s=100, alpha=0.7, edgecolor="w")
            ax.set_xlabel(f'{variable} Promedio', fontsize=12)
            ax.set_ylabel('Calificación', fontsize=12)
            ax.set_title(f'Relación entre {variable} Promedio y Calificaciones', fontsize=16)
            ax.grid(True)

        self.current_canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        self.current_canvas.draw()
        self.current_canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)
        plt.close(fig)

    def plot_rendimiento_academico(self):
        self.clear_canvas()
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=grades_data['Student'], y=grades_data['Grade'], hue=grades_data['Student'], dodge=False, palette='viridis', ax=ax)
        ax.set_xlabel('Estudiantes', fontsize=12)
        ax.set_ylabel('Calificación', fontsize=12)
        ax.set_title('Rendimiento Académico de los Estudiantes en los Exámenes', fontsize=16)
        ax.grid(True)
        self.current_canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        self.current_canvas.draw()
        self.current_canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)
        plt.close(fig)

    def modelado_predictivo(self):
        self.clear_canvas()
        # Verificar si existen los datos requeridos
        required_keys = ['S1_Final_TEMP', 'S1_Final_EDA', 'S1_Final_ACC']
        available_keys = [key for key in required_keys if key in dfs]

        if len(available_keys) < 2:
            messagebox.showerror("Error", "Datos insuficientes: Se necesitan al menos dos variables fisiológicas para proceder con el modelado.")
            return

        # Preparar los datos para el modelado
        try:
            X = pd.concat([dfs[key].iloc[:, 0] for key in available_keys], axis=1).dropna()
            X.columns = available_keys
            grades_data_unique = grades_data.drop_duplicates(subset='Student')
            y = grades_data_unique.set_index('Student').reindex(X.index)['Grade'].dropna()
            X = X.loc[y.index]

            if X.empty or y.empty:
                messagebox.showerror("Error", "Los datos de entrada están vacíos después de la limpieza. No se puede continuar con el modelado.")
                return
        except KeyError as e:
            messagebox.showerror("Error", f"Error al preparar los datos: {e}")
            return

        # Separar datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalizar los datos
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Modelos a utilizar
        models = {
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
            "Linear Regression": LinearRegression(),
            "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
            "Support Vector Regressor": SVR(kernel='rbf')
        }

        results = []

        for model_name, model in models.items():
            # Entrenar el modelo
            model.fit(X_train, y_train)
            # Predicciones
            y_pred = model.predict(X_test)
            # Evaluación del modelo
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            results.append((model_name, mse, r2, mae))

        # Mostrar los resultados en la interfaz
        result_text = "Resultados del Modelado Predictivo:\n"
        for model_name, mse, r2, mae in results:
            result_text += f"\nModelo: {model_name}\nMSE: {mse:.2f}\nR2 Score: {r2:.2f}\nMAE: {mae:.2f}\n"

        messagebox.showinfo("Resultados del Modelado", result_text)

        # Visualización de resultados del último modelo
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pred = models["Random Forest Regressor"].predict(X_test)  # Usar Random Forest para la visualización
        ax.scatter(y_test, y_pred, s=100, alpha=0.7, edgecolor="w")
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
        ax.set_xlabel('Calificaciones Reales', fontsize=12)
        ax.set_ylabel('Calificaciones Predichas', fontsize=12)
        ax.set_title('Real vs Predicción - Calificaciones (Random Forest)', fontsize=16)
        ax.grid(True)
        self.current_canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        self.current_canvas.draw()
        self.current_canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)
        plt.close(fig)

    def tecnicas_estadisticas(self):
        self.clear_canvas()
        participante = simpledialog.askstring("Participante", "Seleccione un participante (S1 - S10, Todos):")
        resumen_data = []
        if participante == "Todos":
            for key, df in dfs.items():
                if not df.empty:
                    resumen_data.append([key, df.iloc[:, 0].mean(), df.iloc[:, 0].median(), df.iloc[:, 0].std(), df.iloc[:, 0].min(), df.iloc[:, 0].max()])
        else:
            for key, df in dfs.items():
                if participante in key and not df.empty:
                    resumen_data.append([key, df.iloc[:, 0].mean(), df.iloc[:, 0].median(), df.iloc[:, 0].std(), df.iloc[:, 0].min(), df.iloc[:, 0].max()])

        # Mostrar tabla de resumen estadístico en el área de gráficos
        tree_frame = Frame(self.graph_frame)
        tree_frame.pack(expand=True, fill='both')

        cols = ('Archivo', 'Media', 'Mediana', 'Desv. Estándar', 'Mínimo', 'Máximo')
        tree = ttk.Treeview(tree_frame, columns=cols, show='headings')

        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=120)

        for item in resumen_data:
            tree.insert("", "end", values=item)

        tree.pack(expand=True, fill='both')

        # Añadir scrollbar
        scrollbar = Scrollbar(tree_frame, orient='vertical', command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side='right', fill='y')

        self.current_treeview = tree_frame

    def plot_desviacion_estandar(self):
        self.clear_canvas()
        variables = ["TEMP", "EDA", "HR", "BVP", "ACC", "IBI"]
        examenes = ["Final", "Midterm 1", "Midterm 2"]
        desviaciones = {variable: [] for variable in variables}

        for variable in variables:
            for examen in examenes:
                desviacion_total = []
                for key, df in dfs.items():
                    if examen in key and variable in key:
                        desviacion_total.append(df.iloc[:, 0].std())
                if desviacion_total:
                    desviaciones[variable].append(np.mean(desviacion_total))
                else:
                    desviaciones[variable].append(0)

        fig, ax = plt.subplots(figsize=(12, 8))
        for variable in variables:
            ax.plot(examenes, desviaciones[variable], label=variable, marker='o', linewidth=2)

        ax.set_xlabel('Examen', fontsize=12)
        ax.set_ylabel('Desviación Estándar Promedio', fontsize=12)
        ax.set_title('Desviación Estándar Comparativa por Examen Médico', fontsize=16)
        ax.legend()
        ax.grid(True)

        # Hacer interactivo
        annotation = ax.annotate('', xy=(0, 0), xytext=(-20, 20), textcoords='offset points',
                                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                                 arrowprops=dict(arrowstyle='->'))
        annotation.set_visible(False)

        def update_annotation(event):
            vis = annotation.get_visible()
            if event.inaxes == ax:
                for line in ax.get_lines():
                    if line.contains(event)[0]:
                        xdata, ydata = line.get_data()
                        ind = line.contains(event)[1]['ind'][0]
                        annotation.xy = (xdata[ind], ydata[ind])
                        text = f"{line.get_label()}\nExamen: {examenes[ind]}\nDesv. Estándar: {ydata[ind]:.2f}"
                        annotation.set_text(text)
                        annotation.set_visible(True)
                        fig.canvas.draw_idle()
                        return
            if vis:
                annotation.set_visible(False)
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect('motion_notify_event', update_annotation)

        self.current_canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        self.current_canvas.draw()
        self.current_canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)
        plt.close(fig)

if __name__ == "__main__":
    root = tk.Tk()
    app = DataAnalysisApp(root)
    root.mainloop()
