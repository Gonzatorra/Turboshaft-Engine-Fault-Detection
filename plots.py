import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from math import pi
import pandas as pd
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix


def plot_distribuciones_outliers(df, df_stats):
    """
    Genera gráficos de densidad para cada columna numérica de un DataFrame y determina los outliers según los límites del IQR.

    Parámetros:
    -----------
        - df (pd.DataFrame): DataFrame con los datos numéricos.
        - df_stats (pd.DataFrame):  DataFrame que contiene estadísticas por columna, incluyendo:
            - 'lower_bound': Límite inferior para outliers (IQR)
            - 'upper_bound': Límite superior para outliers (IQR)
            - 'percent': Porcentaje de outliers detectados
    """
    todas_columnas = df_stats.index.tolist()

    #Si es solo una distribucion, no hacemos subplots
    if len(todas_columnas) == 1:
        fig, ax = plt.subplots(figsize=(16, 8))
        axes = [ax] #Lo metemos en una lista para poder iterar sobre el
    else:
        #La altura de la figura crece dinámicamente
        fig, axes = plt.subplots(len(todas_columnas), 1, figsize=(16, 4 * len(todas_columnas)))
        #Si axes es un array 2D, aplanarlo para que matplotlib pueda graficarlo bien
        if hasattr(axes, 'flatten'):
            axes = axes.flatten() 

    #iteramos sobre cada variable para dibujarla. 
    for i, columna in enumerate(todas_columnas):
        ax = axes[i] #Seleccionamos el eje (gráfico) actual

        #Limites IQR
        lower = df_stats.loc[columna]['lower_bound']
        upper = df_stats.loc[columna]['upper_bound']
        outliers_mask = (df[columna] < lower) | (df[columna] > upper)

        #Density plot
        sns.kdeplot(df[columna], ax=ax, fill=True, color='lightblue', alpha=0.6, linewidth=2)

        #Marcar los outliers
        sns.rugplot(df[columna][outliers_mask], ax=ax, color='red', height=0.05, label='Outliers')

        #Pintar las lineas de limites IQR
        ax.axvline(lower, color='green', linestyle='--', linewidth=2, label=f'Límite inferior: {lower:.2f}')
        ax.axvline(upper, color='orange', linestyle='--', linewidth=2, label=f'Límite superior: {upper:.2f}')

        ax.set_title(f'{columna} ({df_stats.loc[columna]["percent"]:.1f}% outliers)', fontweight='bold', fontsize=12)
        ax.set_xlabel('Valor')
        ax.set_ylabel('Densidad')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    return fig




def plot_correlacion_heatmap(corr_matrix, vmin=-1, vmax=1, title='Matriz de Correlación'):
    """
    Dibuja un heatmap de una matriz de correlación.
    
    Parámetros:
    -----------
    corr_matrix (pd.DataFrame): Matriz de correlación.
    vmin, vmax (float, opcional): Valores mínimo y máximo para el heatmap.
    title (str, opcional): Título del gráfico.
    """

    #Generamos la máscara para el triángulo superior.
    # Como la matriz de correlación es simétrica ocultamos la parte superior
    # np.triu = Triangle Upper
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    figsize=(16,12)
    cmap='RdBu_r'


    fig = plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, 
                mask=mask, 
                annot=True, #escribe el valor numerico dentro de cada celda
                cmap=cmap, #el mapa de colores
                center=0, #fija el color blanco en el centro (para que veamos claro cuando hay y no hay correlacion)
                square=True, #para que la celda no salga rectangular
                fmt='.2f',  #pone los numeros en dos decimales
                cbar_kws={'shrink': .8}, #reduce la barra de leyenda lateral al 80%
                vmin=vmin, #para el limite superior e inferior de la escala de color
                vmax=vmax)
    
    #'pad=20' ainade espacio entre el titulo y el grafico
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    return fig




def plot_confusion_matrix(y_true, y_pred, class_names, title="Matriz de Confusión", labels=None):
    """
    Dibuja una matriz de confusión usando Seaborn heatmap.

    Parámetros:
    -----------
    y_true : array-like
        Valores reales.
    y_pred : array-like
        Valores predichos.
    class_names : list
        Nombres de las clases.
    title : str
        Título del gráfico.
    """
    #Si se pasan labels, forzamos la forma de la matriz (aseguramos que la matriz tenga el tamaino y orden correcto)
    if labels is not None:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
    else:
        #si no hay labels, basado en los valores unicos encontrados.
        cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8,6))
    #fmt='d' es formato 'decimal integer'. O sea, que pone 10 y no 10.00
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Verdadero")
    ax.set_title(title)
    plt.tight_layout()
    return fig

def plot_heatmap_medianas(median_matrix, title="Heatmap de medianas"):
    """
    Dibuja un heatmap de medianas de sensores por fallo con colormap rojo-blanco-azul.

    Parámetros:
    - median_matrix (pd.DataFrame): Matriz de medianas (fallos x sensores).
    - title (str, opcional): Título del gráfico.
    """
    cmap='RdBu_r' #Paleta Divergente (Red-Blue reversed): Rojo para valores positivos altos, azul para negativos, blanco para neutros.
    figsize=(14,8)
    fig = plt.figure(figsize=figsize)


    sns.heatmap(
        median_matrix,
        annot=True, #muestra el valor exacto de la mediana en la celda
        fmt='.2f',
        cmap=cmap,
        center=0, #asegura que el color blanco corresponda al 0 (valor medio/normal)
        linewidths=1, #dibuja lineas blancas entre celdas para separarlas visualmente
        cbar_kws={'label': 'Mediana Normalizada'}, #etiqueta para la barra de color lateral
        vmin=-1, vmax=1 #escala entre 1 y -1
    )
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Sensores', fontsize=12)
    plt.ylabel('Tipo de Fallo', fontsize=12)
    plt.xticks(rotation=45, ha='right')

    return fig




def pie_chart(df, columna, paleta=None, titulo=None):
    """
    Crea un pie chart a partir de una columna de un DataFrame.

    Parámetros:
    - df (pd.DataFrame)
    - columna (str): nombre de la columna a graficar
    - paleta (dict, opcional): {valor: color} o lista de colores
    - figsize: tamaño de la figura
    - titulo: título opcional del gráfico
    """
    figsize=(8,6)
    fig, ax = plt.subplots(figsize=figsize)

    # Conteo de cada valor único
    conteos = df[columna].value_counts()
    labels = conteos.index.tolist()
    sizes = conteos.values

    if paleta is None:
        cmap = plt.get_cmap('tab20') #tab20 tiene 20 colores distintos
        #creamos una lista de colores del tamano de nuestros datos
        paleta = [cmap(i) for i in range(len(labels))]
    
    #manejo de nuestro diccionario de colores
    if isinstance(paleta, dict):
        colors = [paleta[label] for label in labels]

    #Pie chart
    ax.pie(
        sizes, #los datos numericos
        labels=labels,
        autopct='%1.1f%%', #formato del porcentage interno (1 decimal)
        startangle=90, #rotamos el grafico para que el primer trozo empiece a las 12 en punto
        colors=paleta
    )

    #Título
    if titulo is None:
        titulo = f'Distribución de {columna}'
    ax.set_title(titulo, fontsize=14, fontweight='bold')

    return fig




def bar_chart(df, columna, paleta=None, title="Bar Chart", eje_x="Eje x", eje_y="Eje y"):
    """
    Genera un grafico de barras horizontales con etiquetas de datos y porcentajes
    
    """
    fig, ax1 = plt.subplots(figsize=(8,6))

    #Distribución de valores
    distribucion_valores = df[columna].value_counts()

    if paleta == None:
        paleta = plt.get_cmap('tab20') #tab20 tiene 20 colores distintos
    
    #Gráfico de barras
    distribucion_valores.plot(
        kind='barh',
        ax=ax1,
        color=[paleta[f] for f in distribucion_valores.index],
        edgecolor='black'
    )

    ax1.set_title(title, fontweight='bold', fontsize=14)
    ax1.set_xlabel(eje_x)
    ax1.set_ylabel(eje_y)
    ax1.grid(True, alpha=0.3) #rejilla suave vertical para ayudar a medir

    max_val = max(distribucion_valores) #calculamos el valor mas alto y ainadimos un 15% de margen extra a la derecha. lo hacemos para que el texto de las etiquetas no se corte fuera del grafico.
    ax1.set_xlim(0, max_val * 1.15)  #Que no sobresalga la barra
    
    for i, v in enumerate(distribucion_valores):
        porcentaje = v / distribucion_valores.sum() * 100
        #posicion x: justo al final de la barra + un pequeino margen
        #El texto: "Cantidad (Porcentaje%)"
        #Alineacion vertical centrada con la barra
        #Alineacion horizontal a la izquierda (el texto empieza ahi)
        ax1.text(v + 0.01 * max_val, i, f"{v} ({porcentaje:.1f}%)", va='center', ha='left', fontweight='bold')

    
    return fig






def boxplots_por_sensor(df, sensores=None, fallos=None):
    """
    Genera boxplots de cada sensor mostrando la distribución por tipo de fallo.

    Parámetros:
    - df (pd.DataFrame): DataFrame con los datos.
    - sensores (list, opcional): lista de columnas de sensores a graficar. Si es None, usa todas las numéricas excepto 'Fault_Label'.
    - fallos (list, opcional): lista de fallos a considerar. Si es None, usa todos los valores únicos de 'Fault_Label'.

    Devuelve una figura con todos los boxplots.
    """
    if 'Fault_Label' not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'Fault_Label'.")

    # Si se pasan fallos, filtrar el DataFrame
    if fallos is not None and len(fallos) > 0:
        df = df[df['Fault_Label'].isin(fallos)]
    else:
        #sino usamos todos los fallos
        fallos = df['Fault_Label'].unique().tolist()

    # Si se pasan sensores, usar solo esos
    if sensores is not None and len(sensores) > 0:
        sensores = [s for s in sensores if s in df.columns]
    else:
        #sino usamos todos los sensores (todas las columnas numericas excepto la label)
        sensores = [col for col in df.select_dtypes(include='number').columns if col != 'Fault_Label']

    # Crear figura (altura dinamica para que los subplots no se aplasten entre si)
    fig, axes = plt.subplots(len(sensores), 1, figsize=(16, 4 * len(sensores)))
    if len(sensores) == 1:
        axes = [axes]

    #generamos un grafico por sensor.
    for i, sensor in enumerate(sensores):
        #creamos una lista de arrays. Cada elemento de la lista son los valores del sensor para un tipo de fallo específico.
        #box_data = [ [valores_fallo_A], [valores_fallo_B], [valores_fallo_C] ]
        box_data = [df[df['Fault_Label'] == fallo][sensor].values for fallo in fallos]
        axes[i].boxplot(box_data, labels=fallos)
        axes[i].set_ylabel(sensor, fontsize=11)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3, axis='y')
        axes[i].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    fig.suptitle("Distribución de sensores por tipo de fallo", fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    return fig




def radar_plot(fallos, median_matrix, sensores, umbral=0.3, factor=3):
    """
    Radar chart que muestra los fallos y resalta sensores más activados.
    No normaliza los datos; amplifica visualmente las diferencias.

    Parámetros:
    fallos (list): Lista de fallos a comparar
    median_matrix (pd.DataFrame): DataFrame con medianas por fallo (Z-score)
    sensores (list): Lista de sensores/columnas
    umbral (float, opcional): Z-score mínimo para considerar un sensor activado
    factor (float, opcional): Factor para amplificar visualmente los valores
    """

    # Crear figura y ángulos (configuracion del sistema de coordenadas polares)
    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, polar=True)
    #calculamos los angulos para cada sensor
    #Dividimos el circulo (2*pi) en tantos trozos como sensores tengamos
    angles = np.linspace(0, 2*np.pi, len(sensores), endpoint=False).tolist()
    angles += angles[:1]

    #repetimos el primer angulo al final de la lista para que la linea del grafico vuelva al punto de inicio y no quede abierta
    colors = plt.cm.tab10(np.linspace(0, 1, len(fallos)))
    
    #recopilamos todos los datos que vamos a pintar para decidir los limites del grafico.
    todos_valores = []
    for fallo in fallos:
        #Extraemos valores y aplicamos el factor de amplificacion visual
        valores = median_matrix.loc[fallo, sensores].values * factor
        todos_valores.extend(valores)
    
    todos_valores = np.array(todos_valores)
    
    # Usar percentiles para evitar outliers para mejor legibilidad
    y_min = np.percentile(todos_valores, 5)   # Percentil 5
    y_max = np.percentile(todos_valores, 95)  # Percentil 95
    
    # Asegurar un rango mínimo
    if (y_max - y_min) < 1.0:
        centro = (y_max + y_min) / 2
        y_min = centro - 0.5
        y_max = centro + 0.5
    
    ax.set_ylim(y_min, y_max)


    # Dibujar cada fallo
    for idx, fallo in enumerate(fallos):
        #Obtener datos y amplificar
        valores = median_matrix.loc[fallo, sensores].values * factor

        #"Cerrar el círculo" de los datos
        valores_cerrar = np.concatenate((valores, [valores[0]]))

        #dibujamos la linea y el relleno (la "telaraina").
        ax.plot(angles, valores_cerrar, 'o-', color=colors[idx], alpha=0.3, linewidth=2, label=fallo)
        ax.fill(angles, valores_cerrar, color=colors[idx], alpha=0.1)

        #Resaltar puntos criticos
        #recuperamos los valores Z originales (sin multiplicar por 'factor') para evaluar la importancia
        valores_z = median_matrix.loc[fallo, sensores].values
        activados = np.abs(valores_z) > umbral
        
        if activados.any():
            #Creamos un array lleno de NaNs (vacio) y ponemos valores solo donde 'activados' es True
            puntos_y = np.where(activados, valores, np.nan)
            #Cerramos el ciclo tambien para estos puntos (aunque sean sueltos, por consistencia de indices)
            puntos_y = np.concatenate((puntos_y, [puntos_y[0]]))
            #Pintamos solo los puntos que superan el umbral con circulos mas grandes y solidos.
            ax.plot(angles, puntos_y, 'o', color=colors[idx], 
                   markersize=10, alpha=0.9)
            
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(sensores, fontsize=10)
    

    ax.set_yticklabels([])       # Quita los números del eje Y
    
    ax.set_title('RADAR CHART: Sensores activados por fallo', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.0,1.0))

    return fig
    



def timeline_por_tipo_fallo(df, sensores, fallo=None, n_ejemplos=3):
    """
    Muestra ejemplos aleatorios de un tipo de fallo para visualizar su evolución temporal.

    Parámetros:
        - df (pd.DataFrame): DataFrame original con los valores de los sensores
        - sensores (list): Lista de sensores/columnas
        - fallo (str, opcional): Fallo a evaluar
        - n_ejemplos (int, opcional): Cantidad de fallos a visualizar
    """
    # Filtrar por tipo de fallo
    df_fallo = df[df['Fault_Label'] == fallo].copy()
    
    # Colores
    colores = plt.cm.tab10(np.linspace(0, 1, len(sensores)))
    color_dict = {sensor: color for sensor, color in zip(sensores, colores)}
    
    # Seleccionar N ejemplos aleatorios
    ejemplos = df_fallo.sample(min(n_ejemplos, len(df_fallo)))
    
    # Para cada ejemplo, mostrar ventana temporal
    fig, axes = plt.subplots(len(ejemplos), 1, figsize=(16, 4*len(ejemplos)))
    #Pasamos axes a lista si solo hay 1 ejemplo (para que el bucle no falle)
    if len(ejemplos) == 1:
        axes = [axes]
    
    #procesamos cada evento de fallo seleccionado
    for idx, (_, ejemplo) in enumerate(ejemplos.iterrows()):
        ax = axes[idx]
        #fecha/hora exacta del fallo
        tiempo_ejemplo = ejemplo['Timestamp']
                
        # Tomar ventana de tiempo (5 horas antes, 1 después)
        inicio = tiempo_ejemplo - pd.Timedelta(hours=5)
        fin = tiempo_ejemplo + pd.Timedelta(hours=1)
        
        ventana = df[(df['Timestamp'] >= inicio) & (df['Timestamp'] <= fin)].copy()
        ventana = ventana.sort_values('Timestamp')
        
        if len(ventana) == 0:
            ax.text(0.5, 0.5, 'Sin datos en esta ventana', ha='center', va='center')
            continue
        
        # Crear eje X de horas relativas (es decir, pasamos de fecha absoluta a horas relativas al fallo, ej -2.5horas)
        ventana['hora_relativa'] = (ventana['Timestamp'] - tiempo_ejemplo).dt.total_seconds() / 3600
        
        # Graficar cada sensor
        for sensor in sensores:
            ax.plot(ventana['hora_relativa'], ventana[sensor], 
                   color=color_dict[sensor], linewidth=1.5, alpha=0.7, label=sensor)
        
        # Marcar momento del fallo
        ax.axvline(x=0, color='red', linestyle='--', linewidth=3, alpha=0.8)
        ax.axvspan(0, ventana['hora_relativa'].max(), alpha=0.1, color='red')
        
        # Configurar
        ax.set_xlabel('Horas relativas al fallo', fontsize=11)
        ax.set_ylabel('Z-score', fontsize=11)
        ax.set_title(f'Ejemplo {idx+1}: {fallo} a las {tiempo_ejemplo}', fontsize=13)
        ax.grid(True, alpha=0.3)
        
        # Leyenda solo en el primero
        if idx == 0:
            ax.legend(loc='upper right', ncol=3, fontsize=8)
    
    plt.suptitle(f'Análisis de {fallo} - {len(ejemplos)} ejemplos aleatorios', 
                fontsize=16, fontweight='bold', y=1.02)
    
    
    return fig, ejemplos



def plot_cambio_5h(df, sensores):
    """
    Muestra la diferencia de cada sensor 5 horas antes de dar fallo.

    Parámetros:
        - df (pd.DataFrame): DataFrame original con los valores de los sensores
        - sensores (list): Lista de sensores/columnas
    """

    # Obtener fallos
    fallos = [f for f in df['Fault_Label'].unique() if f != 'Normal']
    
    # Ordenar por tiempo
    df_sorted = df.sort_values('Timestamp').copy()
    
    # Referencia Normal (es decir, cual es el valor medio del sensor cuando la maquina funciona bien)
    referencia = df[df['Fault_Label'] == 'Normal'][sensores].mean()
    
    # Para cada fallo, calcular promedio 5h antes
    resultados = []
    
    for fallo in fallos:
        # Encontrar instancias de este fallo (cuando sucedio)
        indices = df_sorted[df_sorted['Fault_Label'] == fallo].index
        
        medias_5h = []
        
        for idx in indices:
            # Tomar 5 horas antes
            inicio = max(0, idx - 5)
            ventana = df_sorted.iloc[inicio:idx]
            
            # Solo si hay al menos 5 horas y son Normales
            if len(ventana) >= 5 and (ventana['Fault_Label'] == 'Normal').all():
                media = ventana[sensores].mean()
                medias_5h.append(media.values)
        
        if medias_5h:
            # Promediar todas las secuencias
            media_promedio = np.array(medias_5h).mean(axis=0)
            # Calcular diferencia vs Normal
            diferencia = media_promedio - referencia.values
            resultados.append((fallo, diferencia, len(medias_5h)))
    
    if not resultados:
        return
    
    # Crear DataFrame
    cambios_df = pd.DataFrame(
        {fallo: diff for fallo, diff, _ in resultados},
        index=sensores
    ).T
    
    # Añadir conteo
    conteos = {fallo: n for fallo, _, n in resultados}
    cambios_df['n'] = cambios_df.index.map(conteos)
    
    # Ordenar por cambio absoluto promedio
    cambios_df['cambio_total'] = cambios_df[sensores].sum(axis=1)
    cambios_df['cambio_abs'] = cambios_df['cambio_total'].abs()
    cambios_df = cambios_df.sort_values('cambio_abs', ascending=False)
    
    # Gráfico simple con barras apiladas
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = range(len(cambios_df))
    
    # Colores por sensor
    colores = plt.cm.tab10(np.linspace(0, 1, len(sensores)))
    
    # Reordenar sensores para mejor visualización
    # Calcular cuál sensor tiene mayor contribución absoluta promedio
    contribuciones = pd.DataFrame({
        sensor: cambios_df[sensor].abs().mean() 
        for sensor in sensores
    }, index=['contrib']).T
    contribuciones = contribuciones.sort_values('contrib', ascending=False)
    sensores_ordenados = contribuciones.index.tolist()
    
    # Para cada sensor, agregar su contribución (APILADAS)
    # Primero separar positivos y negativos
    bottom_pos = np.zeros(len(cambios_df))  # Para valores positivos
    bottom_neg = np.zeros(len(cambios_df))  # Para valores negativos
    
    for sensor in sensores_ordenados:
        valores = cambios_df[sensor].values
        
        # Para valores positivos (arriba de cero)
        positivos = np.where(valores > 0, valores, 0)
        if np.any(positivos > 0):
            ax.bar(x_pos, positivos, bottom=bottom_pos, 
                  color=colores[sensores.index(sensor)], alpha=0.8,
                  edgecolor='black', linewidth=0.5)
            bottom_pos += positivos
        
        # Para valores negativos (abajo de cero)
        negativos = np.where(valores < 0, valores, 0)
        if np.any(negativos < 0):
            # Para negativos, el bottom empieza en 0 y va hacia abajo
            # Así que necesitamos acumular hacia abajo
            ax.bar(x_pos, negativos, bottom=bottom_neg, 
                  color=colores[sensores.index(sensor)], alpha=0.8,
                  edgecolor='black', linewidth=0.5)
            bottom_neg += negativos  # Esto acumula valores negativos (más negativos)
    
    # Línea en y=0
    ax.axhline(y=0, color='black', linewidth=1.5, zorder=0)
    
    # Configurar
    ax.set_xticks(x_pos)
    
    # Etiquetas con conteo
    etiquetas = []
    for fallo in cambios_df.index:
        n = cambios_df.loc[fallo, 'n']
        etiquetas.append(f'{fallo}\n(n={n})')
    
    ax.set_xticklabels(etiquetas, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Z-score (5h antes vs Normal)', fontsize=12)
    ax.set_title('Cambios 5 horas antes de cada fallo', fontsize=14, fontweight='bold')
    
    # Leyenda con sensores en el orden de visualización
    legend_elements = [
        Patch(facecolor=colores[sensores.index(sensor)], alpha=0.8, label=sensor)
        for sensor in sensores_ordenados
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1.0))
    
    ax.grid(True, alpha=0.3, axis='y')

    return fig
    
