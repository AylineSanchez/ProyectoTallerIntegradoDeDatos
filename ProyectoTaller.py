# Importación de bibliotecas necesarias
import pandas as pd  # Para manejar datos en estructuras tipo DataFrame
import numpy as np  # Para realizar cálculos matemáticos y operaciones con arrays
import os  # Para interactuar con el sistema operativo, como rutas de archivos
import matplotlib.pyplot as plt  # Para generar gráficos
import seaborn as sns  # Para mejorar la visualización de gráficos con estilos predefinidos
from sklearn.model_selection import train_test_split, cross_val_score  # Para dividir datos y validar modelos
from sklearn.preprocessing import StandardScaler  # Para escalar características (normalización)
from sklearn.ensemble import RandomForestRegressor  # Modelo de regresión basado en árboles de decisión
from sklearn.linear_model import LinearRegression  # Modelo de regresión lineal
from sklearn.neighbors import KNeighborsRegressor  # Regresión basada en vecinos más cercanos
from sklearn.svm import SVR  # Soporte vectorial para regresión
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error  # Métricas de evaluación
from sklearn.neural_network import MLPRegressor  # Regresor de redes neuronales multicapa
import tkinter as tk  # Biblioteca para crear interfaces gráficas de usuario (GUI)
from tkinter import ttk, messagebox, simpledialog, Scrollbar, Canvas, Frame  # Elementos GUI específicos
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Para integrar gráficos Matplotlib en Tkinter
from tkinter import Label, StringVar  # Componentes adicionales de Tkinter

# Configuración del estilo para gráficos de Seaborn
sns.set(style="whitegrid")  # Estilo blanco con rejilla para los gráficos

# Obtener la ruta del directorio donde se encuentra este script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Definir rutas relativas a partir del directorio actual
grades_file_path = os.path.join(current_dir, "Data", "StudentGrades.txt")  # Ruta al archivo de calificaciones
root_dir = os.path.join(current_dir, "Data", "Data")  # Ruta al directorio de datos

# Imprimir las rutas generadas para verificar que son correctas
print("Ruta de calificaciones:", grades_file_path)
print("Ruta de datos:", root_dir)

# Función para leer y procesar calificaciones desde un archivo de texto
def read_grades(file_path):
    sections = ['MIDTERM 1', 'MIDTERM 2', 'FINAL']  # Secciones de exámenes esperadas
    grades = []  # Lista para almacenar calificaciones procesadas

    current_section = None  # Variable para rastrear la sección actual
    with open(file_path, 'r') as file:  # Abrir el archivo en modo lectura
        for line in file:
            line = line.strip()  # Eliminar espacios en blanco al inicio y final de la línea
            if line.startswith("GRADES"):  # Identificar el encabezado de una sección
                for section in sections:  # Buscar qué sección coincide
                    if section in line.upper():
                        current_section = section
                        break
            elif line and current_section:  # Si hay texto y estamos dentro de una sección
                student_grade = line.split(' - ')  # Separar participante y calificación
                if len(student_grade) == 2:  # Validar formato correcto
                    student, grade = student_grade
                    grades.append({
                        "Participant": student,
                        "Exam": current_section,
                        "Grade": int(grade)  # Convertir calificación a entero
                    })

    return pd.DataFrame(grades)  # Retornar un DataFrame con los datos

# Función para mostrar las calificaciones agrupadas por examen
def mostrar_calificaciones(grades_data):
    for section, grades in grades_data.groupby("Exam"):  # Agrupar por columna 'Exam'
        print(f"\n{section}:")  # Imprimir nombre de la sección
        for _, row in grades.iterrows():  # Iterar sobre las filas del DataFrame
            print(f"  {row['Participant']}: {row['Grade']}")  # Imprimir participante y su calificación

# Leer y mostrar las calificaciones desde el archivo
grades_data = read_grades(grades_file_path)
print(grades_data)

# Crear un diccionario para almacenar DataFrames por archivo
dfs = {}

# Contadores para llevar registro de archivos leídos correctamente y fallidos
successful_reads = 0
failed_reads = 0

# Iterar sobre carpetas de participantes numeradas de S1 a S10
for participant_folder in range(1, 11):  # Rango del 1 al 10
    participant_folder_name = f"S{participant_folder}"  # Nombrar carpeta, e.g., S1
    participant_path = os.path.join(root_dir, participant_folder)  # Ruta completa de la carpeta
    
    # Iterar sobre subcarpetas para exámenes
    for subfolder in ["Final", "Midterm 1", "Midterm 2"]:
        subfolder_path = os.path.join(participant_path, subfolder)  # Ruta de la subcarpeta
        
        # Lista de nombres de archivos esperados en cada subcarpeta
        filenames = ["EDA.csv", "HR.csv", "TEMP.csv", "BVP.csv", "ACC.csv", "IBI.csv"]
        
        # Iterar sobre los nombres de archivos esperados
        for filename in filenames:
            file_path = os.path.join(subfolder_path, filename)  # Ruta completa del archivo
            
            # Comprobar si el archivo existe
            if os.path.isfile(file_path):
                try:
                    # Leer archivo CSV y procesar según sea necesario
                    if filename in ["EDA.csv", "HR.csv", "TEMP.csv", "BVP.csv", "ACC.csv"]:
                        # Leer metadatos de tiempo inicial y frecuencia de muestreo
                        with open(file_path, 'r', encoding='latin-1') as f:
                            initial_time = f.readline().strip()
                            sample_rate = f.readline().strip()
                        # Leer datos omitiendo las primeras dos líneas
                        df = pd.read_csv(file_path, skiprows=2, encoding='latin-1')
                        df.attrs['initial_time'] = initial_time
                        df.attrs['sample_rate'] = sample_rate
                    else:
                        df = pd.read_csv(file_path, encoding='latin-1')
                    
                    # Guardar DataFrame con una clave única en el diccionario
                    key = f"{participant_folder_name}_{subfolder}_{filename.split('.')[0]}"
                    dfs[key] = df

                    # Incrementar contador de éxitos
                    successful_reads += 1
                    print(f"Archivo leído correctamente: {file_path}")

                except Exception as e:
                    # Incrementar contador de errores y mostrar mensaje
                    failed_reads += 1
                    print(f"Error al leer el archivo {file_path}: {e}")
            else:
                # Contabilizar archivos faltantes
                failed_reads += 1
                print(f"El archivo no existe: {file_path}")

# Mostrar resumen del proceso de lectura
print("\nResumen del proceso de lectura:")
print(f"Archivos leídos correctamente: {successful_reads}")
print(f"Archivos fallidos: {failed_reads}")

# Mostrar claves del diccionario que contiene los DataFrames
print("\nArchivos almacenados en el diccionario 'dfs':")
for key, df in dfs.items():
    print(f"\n{key}")

class DataAnalysisApp:
    # Inicializa la aplicación con su ventana principal y configura los elementos básicos
    def __init__(self, root):
        self.root = root  # Ventana principal (Tkinter root)
        self.root.title("Análisis Exploratorio de Datos y Modelado Predictivo")  # Título de la ventana
        self.root.geometry("1200x800")  # Tamaño inicial de la ventana
        self.root.configure(bg='#f0f0f0')  # Color de fondo de la ventana
        self.create_widgets()  # Llama al método para crear los widgets de la interfaz
        self.current_canvas = None  # Variable para almacenar el área de gráficos activa
        self.current_treeview = None  # Variable para almacenar la vista de tabla activa

    def create_widgets(self):
        # Crear un marco para los botones y otro para el área de gráficos
        button_frame = Frame(self.root, bg='#f0f0f0')  # Marco para los botones
        button_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)  # Posiciona el marco a la izquierda
        graph_frame = Frame(self.root, bg='#ffffff')  # Marco para los gráficos
        graph_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)  # Posiciona el marco a la derecha y lo hace expandible
        self.graph_frame = graph_frame  # Almacena la referencia del marco de gráficos

        # Crear botones para cada tipo de funcionalidad de la aplicación
        button_options = [
            ("Tendencias por Participante", self.plot_tendencias),
            ("Datos Contextuales - Rendimiento Académico", self.plot_rendimiento_academico),
            ("Boxplots de Variables Fisiológicas", self.plot_boxplots_variables),
            ("Técnicas Estadísticas para Resumir Hallazgos", self.tecnicas_estadisticas),
            ("Detección de Valores Atípicos", self.plot_outliers_detection),
            ("Modelado Predictivo", self.modelado_predictivo),
            ("Salir", self.root.quit)  # Cierra la aplicación
        ]

        # Crear y configurar cada botón
        for text, command in button_options:
            btn = ttk.Button(button_frame, text=text, command=command)  # Crea el botón
            btn.pack(pady=5, fill=tk.X)  # Posiciona el botón con espacio entre ellos
            btn.configure(style='TButton')  # Aplica un estilo a los botones

    def clear_canvas(self):
        # Borra todos los widgets gráficos dentro del marco de gráficos
        for widget in self.graph_frame.winfo_children():  # Recorre los widgets hijos del marco
            widget.destroy()  # Elimina cada widget
        self.current_canvas = None  # Reinicia el área de gráficos activa
        self.current_treeview = None  # Reinicia la tabla activa

    def plot_tendencias(self):
        self.clear_canvas()  # Limpia el área de gráficos

        # Crear un marco para los menús desplegables y el botón de confirmación
        select_frame = Frame(self.graph_frame, bg='#ffffff')  # Marco para seleccionar opciones
        select_frame.pack(pady=20)  # Espaciado alrededor del marco

        # Variables para almacenar la selección del usuario
        participante_var = StringVar()  # Variable para el participante
        examen_var = StringVar()  # Variable para el examen
        variable_var = StringVar()  # Variable para la variable fisiológica

        # Crear los menús desplegables para seleccionar participante, examen y variable
        ttk.Label(select_frame, text="Seleccione un participante:", background='#ffffff').pack(pady=5)
        participante_combobox = ttk.Combobox(select_frame, textvariable=participante_var)
        participante_combobox['values'] = [f"S{i}" for i in range(1, 11)]  # Opciones: S1 a S10
        participante_combobox.pack()

        ttk.Label(select_frame, text="Seleccione el tipo de examen:", background='#ffffff').pack(pady=5)
        examen_combobox = ttk.Combobox(select_frame, textvariable=examen_var)
        examen_combobox['values'] = ["Final", "Midterm 1", "Midterm 2", "Todos"]  # Tipos de exámenes
        examen_combobox.pack()

        ttk.Label(select_frame, text="Seleccione la variable fisiológica:", background='#ffffff').pack(pady=5)
        variable_combobox = ttk.Combobox(select_frame, textvariable=variable_var)
        variable_combobox['values'] = ["TEMP", "EDA", "HR", "BVP", "ACC", "IBI"]  # Variables fisiológicas
        variable_combobox.pack()

        # Botón para confirmar la selección y generar el gráfico
        def confirmar_seleccion():
            self.clear_canvas()  # Limpia el área de gráficos antes de generar uno nuevo
            participante = participante_var.get()  # Obtiene el participante seleccionado
            examen = examen_var.get()  # Obtiene el examen seleccionado
            variable = variable_var.get()  # Obtiene la variable seleccionada

            # Validar que todos los campos tengan un valor seleccionado
            if not participante or not variable or not examen:
                messagebox.showerror("Error", "Debe seleccionar todos los campos.")  # Muestra un error
                return

            fig, ax = plt.subplots(figsize=(14, 8))  # Crear una figura y eje para el gráfico

            def plot_acc_data(df, label):
                """Graficar datos de acelerómetro (tres ejes: X, Y, Z)."""
                ax.plot(df.index, df.iloc[:, 0], label=f'{label} - X', color='r')
                ax.plot(df.index, df.iloc[:, 1], label=f'{label} - Y', color='g')
                ax.plot(df.index, df.iloc[:, 2], label=f'{label} - Z', color='b')

            def plot_ibi_data(df, label):
                """Graficar datos de intervalos IBI."""
                ax.scatter(df.iloc[:, 0], df.iloc[:, 1], label=f'{label} - IBI', color='m', alpha=0.7)

            def plot_default_data(df, label):
                """Graficar una serie estándar."""
                ax.plot(df.index, df.iloc[:, 0], label=label)

            # Determinar el tipo de gráfico a generar según la selección del usuario
            if examen == "Todos":
                for subfolder in ["Final", "Midterm 1", "Midterm 2"]:
                    key = f"{participante}_{subfolder}_{variable}"  # Clave para buscar los datos
                    if key in dfs:  # Verificar si los datos existen
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

            # Configurar ejes, leyenda y mostrar el gráfico
            ax.set_xlabel('Tiempo', fontsize=12)
            ax.set_ylabel(variable, fontsize=12)
            ax.legend(loc="upper left", bbox_to_anchor=(0, 0))  # Posicionar la leyenda
            self.current_canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)  # Insertar gráfico en Tkinter
            self.current_canvas.draw()  # Dibujar el gráfico
            self.current_canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)  # Mostrar el gráfico
            plt.close(fig)  # Cerrar la figura en Matplotlib

        ttk.Button(select_frame, text="Confirmar Selección", command=confirmar_seleccion).pack(pady=10)  # Botón para confirmar

    def tecnicas_estadisticas(self):  # Método para mostrar técnicas estadísticas de resumen.
        self.clear_canvas()  # Limpia cualquier contenido gráfico actual.

        # Crear un marco para los menús desplegables y el botón de confirmación.
        select_frame = Frame(self.graph_frame, bg='#ffffff')  # Crea un marco dentro del área gráfica.
        select_frame.pack(pady=20)  # Añade un margen vertical al marco.

        # Variable para almacenar la selección del participante.
        participante_var = StringVar()  # Declara una variable StringVar para vincular el valor seleccionado.

        # Combobox para seleccionar el participante.
        ttk.Label(select_frame, text="Seleccione un participante:", background='#ffffff').pack(pady=5)  # Etiqueta descriptiva.
        participante_combobox = ttk.Combobox(select_frame, textvariable=participante_var)  # Combobox para seleccionar participantes.
        participante_combobox['values'] = [f"S{i}" for i in range(1, 11)] + ["Todos"]  # Valores disponibles en el combobox.
        participante_combobox.pack()  # Muestra el combobox.

        # Botón para confirmar la selección y mostrar la tabla estadística.
        def confirmar_seleccion():  # Define la acción al presionar el botón.
            self.clear_canvas()  # Limpia el contenido gráfico actual.
            participante = participante_var.get()  # Obtiene el valor seleccionado del combobox.

            # Validar la entrada del participante.
            if not participante:  # Comprueba que se haya seleccionado un valor.
                messagebox.showerror("Error", "Debe seleccionar un participante.")  # Muestra un mensaje de error.
                return  # Termina la función si no se selecciona nada.

            resumen_data = []  # Lista para almacenar los datos de resumen estadístico.

            # Verificar si se seleccionó "Todos" o un participante específico.
            if participante == "Todos":  # Si se selecciona "Todos", recorre todos los datos.
                for key, df in dfs.items():  # Itera sobre el diccionario de DataFrames.
                    if not df.empty:  # Comprueba que el DataFrame no esté vacío.
                        # Calcula estadísticas básicas y las añade a la lista.
                        resumen_data.append([key, df.iloc[:, 0].mean(), df.iloc[:, 0].median(),
                                            df.iloc[:, 0].std(), df.iloc[:, 0].min(), df.iloc[:, 0].max()])
            else:  # Si se selecciona un participante específico.
                for key, df in dfs.items():  # Itera sobre el diccionario de DataFrames.
                    if participante in key and not df.empty:  # Filtra los datos del participante seleccionado.
                        # Calcula estadísticas básicas y las añade a la lista.
                        resumen_data.append([key, df.iloc[:, 0].mean(), df.iloc[:, 0].median(),
                                            df.iloc[:, 0].std(), df.iloc[:, 0].min(), df.iloc[:, 0].max()])

            # Mostrar tabla de resumen estadístico en el área de gráficos.
            tree_frame = Frame(self.graph_frame)  # Crea un marco para contener la tabla.
            tree_frame.pack(expand=True, fill='both')  # Expande el marco para ocupar todo el espacio.

            # Definir las columnas de la tabla.
            cols = ('Archivo', 'Media', 'Mediana', 'Desv. Estándar', 'Mínimo', 'Máximo')
            tree = ttk.Treeview(tree_frame, columns=cols, show='headings')  # Crea la tabla.

            for col in cols:  # Configura los encabezados de las columnas.
                tree.heading(col, text=col)  # Asigna el texto al encabezado.
                tree.column(col, width=120)  # Define el ancho de cada columna.

            for item in resumen_data:  # Añade cada fila de datos a la tabla.
                tree.insert("", "end", values=item)

            tree.pack(expand=True, fill='both')  # Expande la tabla para ocupar el espacio disponible.

            # Añadir scrollbar.
            scrollbar = Scrollbar(tree_frame, orient='vertical', command=tree.yview)  # Scroll vertical para la tabla.
            tree.configure(yscroll=scrollbar.set)  # Vincula el scrollbar con la tabla.
            scrollbar.pack(side='right', fill='y')  # Posiciona el scrollbar.

            self.current_treeview = tree_frame  # Guarda la referencia al marco actual.

        # Añadir botón para confirmar la selección.
        ttk.Button(select_frame, text="Confirmar Selección", command=confirmar_seleccion).pack(pady=10)  # Crea y muestra el botón.

    def plot_outliers_detection(self):
        # Limpia el área de gráficos previo a dibujar algo nuevo
        self.clear_canvas()

        # Crear un marco contenedor para los menús desplegables y el botón de confirmación
        select_frame = Frame(self.graph_frame, bg='#ffffff')
        select_frame.pack(pady=20)

        # Variables para almacenar las selecciones del usuario: participante y variable fisiológica
        participante_var = StringVar()
        variable_var = StringVar()

        # Etiqueta y menú desplegable para seleccionar un participante
        ttk.Label(select_frame, text="Seleccione un participante:", background='#ffffff').pack(pady=5)
        participante_combobox = ttk.Combobox(select_frame, textvariable=participante_var)
        participante_combobox['values'] = [f"S{i}" for i in range(1, 11)]  # Participantes del S1 al S10
        participante_combobox.pack()

        # Etiqueta y menú desplegable para seleccionar la variable fisiológica
        ttk.Label(select_frame, text="Seleccione la variable fisiológica:", background='#ffffff').pack(pady=5)
        variable_combobox = ttk.Combobox(select_frame, textvariable=variable_var)
        variable_combobox['values'] = ["TEMP", "EDA", "HR", "BVP", "ACC", "IBI"]  # Lista de variables posibles
        variable_combobox.pack()

        # Define el comportamiento del botón de confirmación
        def confirmar_seleccion():
            # Limpia el área de gráficos previo a dibujar algo nuevo
            self.clear_canvas()

            # Obtiene las selecciones del usuario
            participante = participante_var.get()
            variable = variable_var.get()

            # Crear una figura de Matplotlib
            fig, ax = plt.subplots(figsize=(12, 6))
            colors = ['blue', 'green', 'red']  # Colores para diferenciar los exámenes

            # Iterar por las carpetas que representan los diferentes exámenes
            for i, subfolder in enumerate(["Final", "Midterm 1", "Midterm 2"]):
                key = f"{participante}_{subfolder}_{variable}"  # Clave que identifica un dataset
                if key in dfs:  # Verifica si existen datos para esta combinación
                    df = dfs[key]
                    mean_value = df.iloc[:, 0].mean()  # Media de la columna relevante
                    std_value = df.iloc[:, 0].std()  # Desviación estándar de la columna relevante

                    # Identificar outliers usando 3 desviaciones estándar
                    outliers = df[(df.iloc[:, 0] > mean_value + 3 * std_value) | (df.iloc[:, 0] < mean_value - 3 * std_value)]

                    # Graficar valores normales y valores atípicos
                    ax.scatter(df.index, df.iloc[:, 0], label=f'{subfolder} - Normal', color=colors[i], alpha=0.5)
                    ax.scatter(outliers.index, outliers.iloc[:, 0], label=f'{subfolder} - Outliers', color='black', marker='x')

            # Personalización del gráfico
            ax.set_title(f'Detección de Valores Atípicos en {variable} para {participante}')
            ax.set_xlabel('Tiempo')
            ax.set_ylabel(variable)
            ax.legend()

            # Mostrar el gráfico en el canvas de Tkinter
            self.current_canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
            self.current_canvas.draw()
            self.current_canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)
            plt.close(fig)  # Cierra la figura para liberar memoria

        # Botón para confirmar la selección del usuario
        ttk.Button(select_frame, text="Confirmar Selección", command=confirmar_seleccion).pack(pady=10)

    def extract_features(self, df):
        # Calcula estadísticas básicas sobre la columna principal del DataFrame
        mean = df.iloc[:, 0].mean()
        std = df.iloc[:, 0].std()
        max_val = df.iloc[:, 0].max()
        min_val = df.iloc[:, 0].min()

        # Calcula diferencias absolutas entre valores consecutivos (para detectar cambios bruscos)
        abs_diff = df.iloc[:, 0].diff().abs()
        max_abs_diff = abs_diff.max()
        mean_abs_diff = abs_diff.mean()

        # Retorna las características calculadas como una serie de pandas
        return pd.Series({
            "mean": mean,
            "std": std,
            "max": max_val,
            "min": min_val,
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff
        })


    def modelado_predictivo(self):
        # Limpiar el lienzo actual donde se mostrarán los gráficos
        self.clear_canvas()

        # Verificar si existen los datos requeridos en dfs (diccionario de datos)
        available_keys = [key for key in dfs if "TEMP" in key or "EDA" in key or "BVP" in key]

        # Verificar que haya al menos dos variables fisiológicas disponibles
        if len(available_keys) < 2:
            # Mostrar un mensaje de error si no hay suficientes variables
            messagebox.showerror("Error", "Datos insuficientes: Se necesitan al menos dos variables fisiológicas para proceder con el modelado.")
            return

        try:
            # Lista para almacenar las características extraídas de los datos
            features = []
            for key in available_keys:
                # Acceder a los datos del participante, examen y variable
                df = dfs[key]
                participant, exam, variable = key.split('_')
                
                # Extraer las características estadísticas del DataFrame
                feature_row = self.extract_features(df)
                feature_row["Participant"] = participant  # Añadir el participante
                feature_row["Exam"] = exam.upper()  # Asegurar que el examen esté en mayúsculas
                feature_row["Variable"] = variable  # Añadir la variable fisiológica
                features.append(feature_row)  # Añadir las características a la lista

            # Convertir las características extraídas en un DataFrame
            df_features = pd.DataFrame(features)

            print(df_features)  # Imprimir el DataFrame de características extraídas

            # Eliminar ceros a la izquierda en 'grades_data' para que coincidan con el formato de 'dfs'
            grades_data['Participant'] = grades_data['Participant'].apply(lambda x: f"S{int(x[1:]):d}")  # Convierte 'S01' -> 'S1', 'S02' -> 'S2'
            grades_data['Exam'] = grades_data['Exam'].str.strip()  # Eliminar espacios en blanco de 'Exam'

            print(grades_data)  # Imprimir el DataFrame de calificaciones

            # Fusionar las características fisiológicas con las calificaciones
            merged_data = df_features.merge(grades_data, left_on=['Participant', 'Exam'], right_on=['Participant', 'Exam'], how='inner')

            print(merged_data)  # Imprimir el DataFrame fusionado

            # Verificar si después de la fusión no hay datos vacíos
            if merged_data.empty:
                messagebox.showerror("Error", "No se encontraron datos coincidentes entre las características fisiológicas y las calificaciones.")
                return

            # Definir las características (X) y la variable objetivo (y)
            X = merged_data.drop(columns=['Participant', 'Exam', 'Variable', 'Grade'])  # Eliminar las columnas no numéricas
            y = merged_data['Grade']  # La variable objetivo es la calificación

            # Asegurarse de que X contiene solo valores numéricos
            X = X.select_dtypes(include=[np.number])  # Mantener solo las columnas numéricas

            # Restablecer los índices de X e y para asegurarse de que están bien alineados
            X.reset_index(drop=True, inplace=True)
            y.reset_index(drop=True, inplace=True)

            # Comprobar si los datos de entrada están vacíos después de la limpieza
            if X.empty or y.empty:
                messagebox.showerror("Error", "Los datos de entrada están vacíos después de la limpieza. No se puede continuar con el modelado.")
                return

            # Separar los datos en conjuntos de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Normalizar los datos de entrenamiento y prueba
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Definir los modelos a utilizar en el modelado predictivo
            models = {
                "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
                "Linear Regression": LinearRegression(),
                "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
                "Support Vector Regressor": SVR(kernel='rbf'),
                "Neural Network (MLP)": MLPRegressor(hidden_layer_sizes=(100,), max_iter=5000, learning_rate_init=0.001, random_state=42)
            }

            # Lista para almacenar los resultados de los modelos
            results = []
            fig, axs = plt.subplots(len(models), 1, figsize=(10, 6 * len(models)))  # Crear un gráfico por modelo

            # Entrenamiento y evaluación de cada modelo
            for i, (model_name, model) in enumerate(models.items()):
                # Evaluación usando validación cruzada (5 pliegues) para cada modelo
                scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                model.fit(X_train, y_train)  # Entrenar el modelo
                y_pred = model.predict(X_test)  # Realizar predicciones en el conjunto de prueba

                # Calcular las métricas de error del modelo
                mse = mean_squared_error(y_test, y_pred)  # Error cuadrático medio
                r2 = r2_score(y_test, y_pred)  # R^2 (coeficiente de determinación)
                mae = mean_absolute_error(y_test, y_pred)  # Error absoluto medio
                avg_cv_r2 = scores.mean()  # Promedio de R^2 de la validación cruzada

                # Almacenar los resultados de cada modelo
                results.append((model_name, mse, r2, mae, avg_cv_r2))

                # Crear un gráfico Real vs Predicción para cada modelo
                ax = axs[i]
                ax.scatter(y_test, y_pred, s=100, alpha=0.7, edgecolor="w")  # Scatter plot de valores reales vs predichos
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)  # Línea de referencia
                ax.set_xlabel('Calificaciones Reales', fontsize=12)
                ax.set_ylabel('Calificaciones Predichas', fontsize=12)
                ax.set_title(f'Real vs Predicción - {model_name}', fontsize=16)
                ax.grid(True)

                # Guardar el gráfico como imagen
                fig.savefig(f"{model_name.replace(' ', '_')}_real_vs_pred.png")

            # Mostrar los resultados del modelado en un cuadro de mensaje
            result_text = "Resultados del Modelado Predictivo:\n"
            for model_name, mse, r2, mae, avg_cv_r2 in results:
                result_text += f"\nModelo: {model_name}\nMSE: {mse:.2f}\nR2 Score: {r2:.2f}\nMAE: {mae:.2f}\nCV R2 Score: {avg_cv_r2:.2f}\n"

            messagebox.showinfo("Resultados del Modelado", result_text)

            # Mostrar los gráficos en la interfaz gráfica
            self.current_canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
            self.current_canvas.draw()
            self.current_canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)
            plt.close(fig)  # Cerrar el gráfico una vez que se haya mostrado

        except KeyError as e:
            # Capturar errores si hay problemas con las claves en los datos
            messagebox.showerror("Error", f"Error al preparar los datos: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = DataAnalysisApp(root)
    root.mainloop()
