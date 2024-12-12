import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neural_network import MLPRegressor
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, Scrollbar, Canvas, Frame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Label
from tkinter import StringVar
import os

# Configuración de estilo para gráficos
sns.set(style="whitegrid")
# Obtener la ruta del directorio actual
current_dir = os.path.dirname(os.path.abspath(__file__))

# Rutas relativas basadas en el directorio actual
grades_file_path = os.path.join(current_dir, "Data", "StudentGrades.txt")
root_dir = os.path.join(current_dir, "Data", "Data")

# Verificar las rutas generadas
print("Ruta de calificaciones:", grades_file_path)
print("Ruta de datos:", root_dir)

# Leer las calificaciones y organizar los datos correctamente
def read_grades(file_path):
    sections = ['MIDTERM 1', 'MIDTERM 2', 'FINAL']
    grades = []

    current_section = None
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("GRADES"):
                # Identificar la sección
                for section in sections:
                    if section in line.upper():
                        current_section = section
                        break
            elif line and current_section:
                # Procesar las calificaciones de cada estudiante
                student_grade = line.split(' - ')
                if len(student_grade) == 2:
                    student, grade = student_grade
                    grades.append({
                        "Participant": student,
                        "Exam": current_section,
                        "Grade": int(grade)
                    })

    return pd.DataFrame(grades)

def mostrar_calificaciones(grades_data):
    for section, grades in grades_data.groupby("Exam"):
        print(f"\n{section}:")
        for _, row in grades.iterrows():
            print(f"  {row['Participant']}: {row['Grade']}")

# Leer y mostrar las calificaciones
grades_data = read_grades(grades_file_path)
print(grades_data)

# Crear el diccionario para almacenar los DataFrames
dfs = {}

# Contadores para llevar registro de los archivos leídos y los fallidos
successful_reads = 0
failed_reads = 0

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

                    # Incrementar contador de lecturas exitosas
                    successful_reads += 1
                    print(f"Archivo leído correctamente: {file_path}")

                except Exception as e:
                    # Incrementar contador de lecturas fallidas e imprimir el error
                    failed_reads += 1
                    print(f"Error al leer el archivo {file_path}: {e}")
            else:
                # Archivo no encontrado
                failed_reads += 1
                print(f"El archivo no existe: {file_path}")

# Mostrar resultados del proceso de lectura
print("\nResumen del proceso de lectura:")
print(f"Archivos leídos correctamente: {successful_reads}")
print(f"Archivos fallidos: {failed_reads}")

# Mostrar las claves almacenadas en el diccionario dfs
print("\nArchivos almacenados en el diccionario 'dfs':")
for key, df in dfs.items():
    print(f"\n{key}")

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
            ("Datos Contextuales - Rendimiento Académico", self.plot_rendimiento_academico),
            ("Boxplots de Variables Fisiológicas", self.plot_boxplots_variables),
            ("Técnicas Estadísticas para Resumir Hallazgos", self.tecnicas_estadisticas),
            ("Detección de Valores Atípicos", self.plot_outliers_detection),
            ("Modelado Predictivo", self.modelado_predictivo),
            ("Salir", self.root.quit)
        ]

        for text, command in button_options:
            btn = ttk.Button(button_frame, text=text, command=command)
            btn.pack(pady=5, fill=tk.X)
            btn.configure(style='TButton')

    def clear_canvas(self):
        # Eliminar todos los elementos gráficos dentro del área de gráficos
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        self.current_canvas = None
        self.current_treeview = None

    def plot_tendencias(self):
        self.clear_canvas()

        # Crear un marco para los menús desplegables y el botón de confirmación
        select_frame = Frame(self.graph_frame, bg='#ffffff')
        select_frame.pack(pady=20)

        # Variables para almacenar la selección
        participante_var = StringVar()
        examen_var = StringVar()
        variable_var = StringVar()

        # Combobox para seleccionar el participante
        ttk.Label(select_frame, text="Seleccione un participante:", background='#ffffff').pack(pady=5)
        participante_combobox = ttk.Combobox(select_frame, textvariable=participante_var)
        participante_combobox['values'] = [f"S{i}" for i in range(1, 11)]
        participante_combobox.pack()

        # Combobox para seleccionar el examen
        ttk.Label(select_frame, text="Seleccione el tipo de examen:", background='#ffffff').pack(pady=5)
        examen_combobox = ttk.Combobox(select_frame, textvariable=examen_var)
        examen_combobox['values'] = ["Final", "Midterm 1", "Midterm 2", "Todos"]
        examen_combobox.pack()

        # Combobox para seleccionar la variable fisiológica
        ttk.Label(select_frame, text="Seleccione la variable fisiológica:", background='#ffffff').pack(pady=5)
        variable_combobox = ttk.Combobox(select_frame, textvariable=variable_var)
        variable_combobox['values'] = ["TEMP", "EDA", "HR", "BVP", "ACC", "IBI"]
        variable_combobox.pack()

        # Botón para confirmar la selección y mostrar el gráfico
        def confirmar_seleccion():
            self.clear_canvas()
            participante = participante_var.get()
            examen = examen_var.get()
            variable = variable_var.get()

            # Validar que se seleccionaron los valores
            if not participante or not variable or not examen:
                messagebox.showerror("Error", "Debe seleccionar todos los campos.")
                return

            fig, ax = plt.subplots(figsize=(14, 8))

            def plot_acc_data(df, label):
                """Graficar los datos de ACC (tres ejes)."""
                ax.plot(df.index, df.iloc[:, 0], label=f'{label} - X', color='r')
                ax.plot(df.index, df.iloc[:, 1], label=f'{label} - Y', color='g')
                ax.plot(df.index, df.iloc[:, 2], label=f'{label} - Z', color='b')

            def plot_ibi_data(df, label):
                """Graficar los datos de IBI (intervalos)."""
                ax.scatter(df.iloc[:, 0], df.iloc[:, 1], label=f'{label} - IBI', color='m', alpha=0.7)

            def plot_default_data(df, label):
                """Graficar una serie estándar."""
                ax.plot(df.index, df.iloc[:, 0], label=label)

            # Determinar el tipo de gráfico a generar según la variable seleccionada
            if examen == "Todos":
                for subfolder in ["Final", "Midterm 1", "Midterm 2"]:
                    key = f"{participante}_{subfolder}_{variable}"
                    if key in dfs:
                        df = dfs[key]
                        if variable == "ACC":
                            plot_acc_data(df, subfolder)
                        elif variable == "IBI":
                            plot_ibi_data(df, subfolder)
                        else:
                            plot_default_data(df, subfolder)
                ax.set_title(f'Tendencia de {variable} para {participante} en todos los exámenes')
            else:
                key = f"{participante}_{examen}_{variable}"
                if key in dfs:
                    df = dfs[key]
                    if variable == "ACC":
                        plot_acc_data(df, examen)
                    elif variable == "IBI":
                        plot_ibi_data(df, examen)
                    else:
                        plot_default_data(df, examen)
                ax.set_title(f'Tendencia de {variable} para {participante} en el {examen}')

            ax.set_xlabel('Tiempo', fontsize=12)
            ax.set_ylabel(variable, fontsize=12)
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
            self.current_canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
            self.current_canvas.draw()
            self.current_canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)
            plt.close(fig)

        ttk.Button(select_frame, text="Confirmar Selección", command=confirmar_seleccion).pack(pady=10)

    def tecnicas_estadisticas(self):
        self.clear_canvas()

        # Crear un marco para los menús desplegables y el botón de confirmación
        select_frame = Frame(self.graph_frame, bg='#ffffff')
        select_frame.pack(pady=20)

        # Variable para almacenar la selección del participante
        participante_var = StringVar()

        # Combobox para seleccionar el participante
        ttk.Label(select_frame, text="Seleccione un participante:", background='#ffffff').pack(pady=5)
        participante_combobox = ttk.Combobox(select_frame, textvariable=participante_var)
        participante_combobox['values'] = [f"S{i}" for i in range(1, 11)] + ["Todos"]
        participante_combobox.pack()

        # Botón para confirmar la selección y mostrar la tabla estadística
        def confirmar_seleccion():
            self.clear_canvas()
            participante = participante_var.get()

            # Validar la entrada del participante
            if not participante:
                messagebox.showerror("Error", "Debe seleccionar un participante.")
                return

            resumen_data = []

            # Verificar si se seleccionó "Todos" o un participante específico
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

        # Añadir botón para confirmar la selección
        ttk.Button(select_frame, text="Confirmar Selección", command=confirmar_seleccion).pack(pady=10)

    def plot_rendimiento_academico(self):
        self.clear_canvas()

        # Filtrar y preparar datos de calificaciones para el gráfico
        df_grades = grades_data[['Participant', 'Grade', 'Exam']]

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x='Participant', y='Grade', hue='Exam', data=df_grades, palette='viridis', ax=ax)

        ax.set_xlabel('Participantes', fontsize=12)
        ax.set_ylabel('Calificaciones', fontsize=12)
        ax.set_title('Calificaciones de los Participantes en los Exámenes', fontsize=16)
        ax.legend(title="Exámenes")
        ax.grid(True)

        self.current_canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        self.current_canvas.draw()
        self.current_canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)
        plt.close(fig)


    def plot_boxplots_variables(self):
        self.clear_canvas()

        # Crear un marco para los menús desplegables y el botón de confirmación
        select_frame = Frame(self.graph_frame, bg='#ffffff')
        select_frame.pack(pady=20)

        # Variable para almacenar la selección de la variable
        variable_var = StringVar()

        # Combobox para seleccionar la variable fisiológica
        ttk.Label(select_frame, text="Seleccione la variable fisiológica:", background='#ffffff').pack(pady=5)
        variable_combobox = ttk.Combobox(select_frame, textvariable=variable_var)
        variable_combobox['values'] = ["TEMP", "EDA", "HR", "BVP", "ACC", "IBI"]
        variable_combobox.pack()

        # Botón para confirmar la selección y mostrar el gráfico de boxplot
        def confirmar_seleccion():
            self.clear_canvas()
            variable = variable_var.get()
            data = []

            for key in dfs.keys():
                if key.endswith(variable):
                    df = dfs[key]
                    data.extend(df.iloc[:, 0].tolist())

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=data, ax=ax)
            ax.set_title(f'Boxplot de {variable} para todos los exámenes y participantes')
            ax.set_ylabel(variable)
            self.current_canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
            self.current_canvas.draw()
            self.current_canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)
            plt.close(fig)

        ttk.Button(select_frame, text="Confirmar Selección", command=confirmar_seleccion).pack(pady=10)

    def plot_outliers_detection(self):
        self.clear_canvas()

        # Crear un marco para los menús desplegables y el botón de confirmación
        select_frame = Frame(self.graph_frame, bg='#ffffff')
        select_frame.pack(pady=20)

        # Variables para almacenar la selección del participante y la variable
        participante_var = StringVar()
        variable_var = StringVar()

        # Combobox para seleccionar el participante
        ttk.Label(select_frame, text="Seleccione un participante:", background='#ffffff').pack(pady=5)
        participante_combobox = ttk.Combobox(select_frame, textvariable=participante_var)
        participante_combobox['values'] = [f"S{i}" for i in range(1, 11)]
        participante_combobox.pack()

        # Combobox para seleccionar la variable fisiológica
        ttk.Label(select_frame, text="Seleccione la variable fisiológica:", background='#ffffff').pack(pady=5)
        variable_combobox = ttk.Combobox(select_frame, textvariable=variable_var)
        variable_combobox['values'] = ["TEMP", "EDA", "HR", "BVP", "ACC", "IBI"]
        variable_combobox.pack()

        # Botón para confirmar la selección y mostrar el gráfico de outliers
        def confirmar_seleccion():
            self.clear_canvas()
            participante = participante_var.get()
            variable = variable_var.get()

            fig, ax = plt.subplots(figsize=(12, 6))
            colors = ['blue', 'green', 'red']

            for i, subfolder in enumerate(["Final", "Midterm 1", "Midterm 2"]):
                key = f"{participante}_{subfolder}_{variable}"
                if key in dfs:
                    df = dfs[key]
                    mean_value = df.iloc[:, 0].mean()
                    std_value = df.iloc[:, 0].std()

                    outliers = df[(df.iloc[:, 0] > mean_value + 3 * std_value) | (df.iloc[:, 0] < mean_value - 3 * std_value)]

                    ax.scatter(df.index, df.iloc[:, 0], label=f'{subfolder} - Normal', color=colors[i], alpha=0.5)
                    ax.scatter(outliers.index, outliers.iloc[:, 0], label=f'{subfolder} - Outliers', color='black', marker='x')

            ax.set_title(f'Detección de Valores Atípicos en {variable} para {participante}')
            ax.set_xlabel('Tiempo')
            ax.set_ylabel(variable)
            ax.legend()
            self.current_canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
            self.current_canvas.draw()
            self.current_canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)
            plt.close(fig)

        ttk.Button(select_frame, text="Confirmar Selección", command=confirmar_seleccion).pack(pady=10)

    def extract_features(self, df):
        # Características estadísticas
        mean = df.iloc[:, 0].mean()
        std = df.iloc[:, 0].std()
        max_val = df.iloc[:, 0].max()
        min_val = df.iloc[:, 0].min()

        # Cambios bruscos: variación absoluta entre valores consecutivos
        abs_diff = df.iloc[:, 0].diff().abs()
        max_abs_diff = abs_diff.max()
        mean_abs_diff = abs_diff.mean()

        return pd.Series({
            "mean": mean,
            "std": std,
            "max": max_val,
            "min": min_val,
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff
        })

    def modelado_predictivo(self):
        self.clear_canvas()

        # Verificar si existen los datos requeridos en dfs
        available_keys = [key for key in dfs if "TEMP" in key or "EDA" in key or "BVP" in key]
        
        # Verificar que hay datos disponibles
        if len(available_keys) < 2:
            messagebox.showerror("Error", "Datos insuficientes: Se necesitan al menos dos variables fisiológicas para proceder con el modelado.")
            return
        
        try:
            # Lista para almacenar las características extraídas
            features = []
            for key in available_keys:
                df = dfs[key]  # Acceder a los datos fisiológicos
                participant, exam, variable = key.split('_')
                feature_row = self.extract_features(df) 
                feature_row["Participant"] = participant
                feature_row["Exam"] = exam.upper()  # Asegurarse de que el nombre del examen esté en mayúsculas
                feature_row["Variable"] = variable
                features.append(feature_row)
            
            # Convertir las características extraídas en un DataFrame
            df_features = pd.DataFrame(features)

            print(df_features)

            # Eliminar ceros a la izquierda en grades_data para que coincidan con el formato de dfs
            grades_data['Participant'] = grades_data['Participant'].apply(lambda x: f"S{int(x[1:]):d}")  # Convierte 'S01' -> 'S1', 'S02' -> 'S2'
            grades_data['Exam'] = grades_data['Exam'].str.strip()

            print(grades_data)

            # Fusionar las características fisiológicas con las calificaciones
            merged_data = df_features.merge(grades_data, left_on=['Participant', 'Exam'], right_on=['Participant', 'Exam'], how='inner')

            print(merged_data)

            # Verificar que no haya datos vacíos después de la fusión
            if merged_data.empty:
                messagebox.showerror("Error", "No se encontraron datos coincidentes entre las características fisiológicas y las calificaciones.")
                return
            
            # Definir las características (X) y la variable objetivo (y)
            X = merged_data.drop(columns=['Participant', 'Exam', 'Variable', 'Grade'])  # Eliminar columnas no numéricas
            y = merged_data['Grade']

            # Asegurarse de que X contiene solo valores numéricos
            X = X.select_dtypes(include=[np.number])  # Mantener solo las columnas numéricas

            # Restablecer índices para asegurarse de que los datos estén bien alineados
            X.reset_index(drop=True, inplace=True)
            y.reset_index(drop=True, inplace=True)

            # Comprobar si los datos de entrada están vacíos después de la limpieza
            if X.empty or y.empty:
                messagebox.showerror("Error", "Los datos de entrada están vacíos después de la limpieza. No se puede continuar con el modelado.")
                return

            # Separar los datos en entrenamiento y prueba
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
                "Support Vector Regressor": SVR(kernel='rbf'),
                "Neural Network (MLP)": MLPRegressor(hidden_layer_sizes=(100,), max_iter=5000, learning_rate_init=0.001, random_state=42)
            }

            results = []
            fig, axs = plt.subplots(len(models), 1, figsize=(10, 6 * len(models)))

            # Entrenamiento y evaluación de modelos
            for i, (model_name, model) in enumerate(models.items()):
                scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                avg_cv_r2 = scores.mean()

                results.append((model_name, mse, r2, mae, avg_cv_r2))

                # Gráfico Real vs Predicción para cada modelo
                ax = axs[i]
                ax.scatter(y_test, y_pred, s=100, alpha=0.7, edgecolor="w")
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
                ax.set_xlabel('Calificaciones Reales', fontsize=12)
                ax.set_ylabel('Calificaciones Predichas', fontsize=12)
                ax.set_title(f'Real vs Predicción - {model_name}', fontsize=16)
                ax.grid(True)

                # Guardar el gráfico como imagen para descarga
                fig.savefig(f"{model_name.replace(' ', '_')}_real_vs_pred.png")

            # Mostrar los resultados
            result_text = "Resultados del Modelado Predictivo:\n"
            for model_name, mse, r2, mae, avg_cv_r2 in results:
                result_text += f"\nModelo: {model_name}\nMSE: {mse:.2f}\nR2 Score: {r2:.2f}\nMAE: {mae:.2f}\nCV R2 Score: {avg_cv_r2:.2f}\n"

            messagebox.showinfo("Resultados del Modelado", result_text)

            # Mostrar los gráficos en la interfaz
            self.current_canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
            self.current_canvas.draw()
            self.current_canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)
            plt.close(fig)

        except KeyError as e:
            messagebox.showerror("Error", f"Error al preparar los datos: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DataAnalysisApp(root)
    root.mainloop()
