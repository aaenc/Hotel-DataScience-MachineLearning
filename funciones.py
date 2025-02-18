# Para temas de ficheros y rutas
import os
# Importar librerías necesarias numpy y pandas
import numpy as np
import pandas as pd
# Importamos librerias para graficar
import matplotlib.pyplot as plt
import seaborn as sns
# Para numeros aleatorios
import random
# Para informacion de paises y sus datos adicionales
import pycountry
import pycountry_convert as pc
# Para visualizacion HTML
from IPython.core.display import HTML


# Calcular filas y columnas para un subplot
def calcular_filas_columnas_subplot(dataset,columnas_por_fila=None,pintarTraza=False):
    if columnas_por_fila==None:
        columnas_por_fila=3
    # Calculamos las columnas y filas de forma dinamica
    ncolumnas=len(dataset.columns)
    if pintarTraza:
        print (f'{ncolumnas} Campos')
    # El numero de filas es la parte entera de dividir los compos entre columnas por fila
    nfilas=(ncolumnas//columnas_por_fila)
    if pintarTraza:
        print (f'{nfilas} Filas')
    # Si el numero de filas es insuficienente le sumamos uno
    #if (nfilas*columnas_por_fila)<ncolumnas:
    if (ncolumnas%columnas_por_fila)>0:
        nfilas+=1
    print (f'{nfilas} Filas x {ncolumnas} Columnas, con {columnas_por_fila} columnas por fila')
    print (f'Generar subplot de {nfilas} Filas x {columnas_por_fila} columnas')
    return nfilas,columnas_por_fila

# Generar un color aleatorio en formato hexadecimal
def random_color():
    return "#{:06x}".format(np.random.randint(0, 0xFFFFFF)) 

# df: dataframe con las columnas a pintar
# nfilas: Nº Filas del SubPlot
# columnas_por_fila: Nº Columnas por fila del SubPlot
# tipoVariable: Tipo de variables a pintar CONTINUA, CATEGORICA
def pintar_subplot_columnas(df,columnas_por_fila=3,tipoVariable="CATEGORICA",tipoGrafica="COUNTPLOT",tituloGeneral="",pintarTraza=False,NumeroMaximoCategorias=100,tamannoFuente=14,bins=20):
    # Queremos pintar N columnas por fila
    columnas_por_fila=3
    if pintarTraza:
        print (f'{columnas_por_fila} Columnas por fila')
    nfilas,columnas_por_fila=calcular_filas_columnas_subplot(df,columnas_por_fila,pintarTraza=pintarTraza)
    # Crear una figura con subplots
    fig, axes = plt.subplots(nfilas, columnas_por_fila, figsize=(38, 30))  # 1 fila, 3 columnas
    # Ajustar márgenes entre subplots
    fig.subplots_adjust(hspace=0.4, wspace=0.4,top=0.85)  # Espacio vertical, horizontal y arriba
    
    grafico_fila=0
    grafico_columna=0
    nGraficosPintados=0
    pintarTraza=False
    nGraficosBreak=999 ## Para pintar un maximo de graficos
    total = len(df)  # Total de filas en el dataset
    if (tipoVariable=="CATEGORICA"):
        # Lista de paletas disponibles en seaborn
        paletas_disponibles = sns.palettes.SEABORN_PALETTES.keys()
    if (tipoVariable=="CONTINUA"):
        # Datos de ejemplo
        data = np.random.randn(1000)
    # # Para qye si pido outLayers entre por la parte de CONTINUA
    # if (tipoGrafica in ("BOXPLOT")):
    #     tipoVariable=="CONTINUA"
    # Por cada columna pintamos el grafico
    for i in df.columns:
        Titulo=f"{i}"
        match tipoVariable:
            case "CONTINUA":
                match tipoGrafica:
                    case "BOXPLOT":
                        ax = sns.boxplot(df[i], vert=True, patch_artist=True, boxprops=dict(facecolor=random_color()),ax=axes[grafico_fila,grafico_columna])
                    case "HISTPLOT":
                        ax = sns.histplot(df[i], kde=True, bins=bins, color=random_color() ,ax=axes[grafico_fila,grafico_columna])
                        # Ajustar límite superior del eje y para dar más espacio
                        #y_limit = ax.get_ylim()  # Obtener límites actuales del eje y
                        #ax.set_ylim([0, y_limit[1] + 15])  # Aumentar el límite superior para dar espacio a los números
                        # Agregar valores encima de cada barra
                        for patch in ax.patches:
                            # Obtener las coordenadas de la barra
                            height = patch.get_height()  # Altura de la barra
                            width = patch.get_width()  # Ancho de la barra
                            x = patch.get_x() + width / 2  # Centro de la barra
                            
                            # Escribir el valor encima de la barra
                            if height > 0:  # Solo mostrar si la altura es mayor que 0
                                ax.text(x, height + 1.5, f"{int(height)}", ha="center", fontsize=tamannoFuente,rotation=90)

                        # # Obtener los valores del histograma sin graficarlo
                        # counts, bins = np.histogram(df[i], bins=bins)  # Ajusta 'auto' o el número de bins deseado
                        # valor_max_redondeado = round(counts.max() * 1.5, 0)  # Aumentar un 50%
                        # # Ajustar el límite del eje Y
                        # ax.set_ylim(0, valor_max_redondeado)
                        # if pintarTraza:
                        #     print(f" Valor Max Redondeado : {valor_max_redondeado}")
                
                    case _:
                        print(f"-> tipoGrafica: '{tipoGrafica}' no definido para tipoVariable: '{tipoVariable}'")
            case "CATEGORICA":
                match tipoGrafica:
                    case "COUNTPLOT":
                        # Vamos acomprobar que no sean valores continuos poniendo un limite de 100 valores diferentes (se me quedaba colgado al meter por error una columna continua)
                        if len(df[i].unique())<NumeroMaximoCategorias:
                            # Elegir una paleta aleatoria
                            paleta_aleatoria = random.choice(list(paletas_disponibles))
                            # Usar la paleta seleccionada aleatoriamente
                            colores_personalizados = sns.color_palette(paleta_aleatoria, n_colors=len(df[i].unique()))
                            # Crear un countplot más estético
                            if pintarTraza:
                                print(f"Fila: {grafico_fila}, Columna {grafico_columna}")
                            ax = sns.countplot(x=df[i], data=df, ax=axes[grafico_fila,grafico_columna],palette=colores_personalizados,hue=df[i])  # Elegir una paleta de colores agradable
                            # Añadir las proporciones encima de las barras
                            for p in ax.patches:
                                count = p.get_height()  # Obtener el conteo de la barra
                                proportion = (count / total)*100  # Calcular la proporción respecto al total
                                ax.annotate(f'{proportion:.2f}%',  # Mostrar la proporción con dos decimales
                                            (p.get_x() + p.get_width() / 2., p.get_height()),  # Posición (X, Y)
                                            ha='center', va='center',  # Alineación
                                            fontsize=10, color='black',  # Tamaño de la fuente y color
                                            xytext=(0, 5), textcoords='offset points')  # Ajuste del texto
                           
                            # Mejorar la visibilidad de las etiquetas
                            plt.xticks(rotation=45, ha='right')  # Rotar las etiquetas para que no se sobrepongan
                            axes[grafico_fila,grafico_columna].set_xlabel(f'Tipo de "{i}"', fontsize=tamannoFuente)
                            axes[grafico_fila,grafico_columna].set_ylabel('Frecuencia', fontsize=tamannoFuente)
                        
                            # Agrupar por la columna df_heart[i] y contar los valores
                            conteos = df.groupby(df[i]).size()
                            # Obtener el valor máximo del count()
                            valor_max_redondeado = round(conteos.max()*1.5,0) 
                            if pintarTraza:
                                print(f" Valor Max Redondeado : {valor_max_redondeado}")
                            axes[grafico_fila,grafico_columna].set_ylim(0, valor_max_redondeado)  # Escala personalizada para este subplot
                            
                            # Crear la leyenda manualmente
                            handles = [plt.Line2D([0], [0], color=col, lw=10) for col in colores_personalizados]
                            labels = df[i].unique()  # Nombres únicos en la columna
                            if len(labels)<10:
                                axes[grafico_fila, grafico_columna].legend(handles, labels, title=Titulo,bbox_to_anchor=(1.05, 1), loc='upper left')
                    case _:
                        print(f"-> tipoGrafica: '{tipoGrafica}' no definido para tipoVariable: '{tipoVariable}'")
            case _:
                print(f"-> tipoVariable: '{tipoVariable}' no definido.")

        # Codigo comun a los dos tipos
        axes[grafico_fila,grafico_columna].set_title(f'Distribución de  "{i}"', fontsize=tamannoFuente, fontweight='bold')
        axes[grafico_fila, grafico_columna].tick_params(axis='x', labelrotation=90)
        
    
        # Aumentamos contador de graficos pintados
        nGraficosPintados+=1
        if pintarTraza:
            print(f"nGraficosPintados : {nGraficosPintados}")
        # Comprobamos si ha cambiado de fila
        grafico_fila_nuevo=nGraficosPintados//columnas_por_fila
        if (grafico_fila_nuevo!=grafico_fila):
            # Reiniciamos el contador de columnas
            grafico_columna=0
            # Calculamos el contador de filas nuevo
            grafico_fila=grafico_fila_nuevo    
            if pintarTraza:
               print(f"     Fila Nueva : {grafico_fila}, Columna Nueva {grafico_columna}")
        else:
            # Aumentamos el contador de columnas
            grafico_columna+=1
        if nGraficosPintados>=nGraficosBreak:    
            print(f"     Salimos por condicion de nGraficosBreak={nGraficosBreak}")
            break
    
    # Elimina subplots vacíos si los hay metodo 2
    for ax in axes.flat[nGraficosPintados:]:
        ax.set_visible(False)
    # Cambiar el color del fondo del gráfico
    #plt.gca().set_facecolor('#2b2b2b')  # Un color gris oscuro
    if (tituloGeneral!=""):
        # Agregar un título general
        fig.suptitle(tituloGeneral, fontsize=22)
        # Ajustar el espacio para que las etiquetas no se corten
        plt.tight_layout(pad=4)  
    else:
        # Ajustar el espacio para que las etiquetas no se corten
        plt.tight_layout()  
    # Mostrar el gráfico
    plt.show()
    return

# dfLista: Lista de dataframes a los que graficar nuclos
def pintar_subplot_nulos(dfLista,tituloGeneral="",pintarTraza=False):
    tituloGeneral="ANALISIS DE CAMPOS NULOS"
    nfilas=1
    columnas_por_fila=len(dfLista)
    # Crear una figura con subplots
    if columnas_por_fila>1:
        fig, axes = plt.subplots(nfilas, columnas_por_fila, figsize=(30, 10))  # 1 fila, 3 columnas
    else:
        fig, axes = plt.subplots(nfilas, columnas_por_fila, figsize=(10, 10))  # 1 fila, 3 columnas
    # Ajustar márgenes entre subplots
    fig.subplots_adjust(hspace=0.4, wspace=0.4)  # Espacio vertical y horizontal
    
    grafico_columna=0
    nGraficosPintados=0
    NumeroDataframes=len(dfLista)
    for name, df in dfLista.items():
        # Porcentaje de valores nulos por columna
        null_percentage = df.isnull().mean() * 100
        # Crear un gráfico de barras para el porcentaje de nulos
        if columnas_por_fila>1:
            ax=sns.barplot(x=null_percentage.index, y=null_percentage.values,color=random_color() ,ax=axes[grafico_columna])
            axes[grafico_columna].set_xlabel(f'Campos de " {name} "', fontsize=12)
            axes[grafico_columna].set_ylabel('Porcentaje NULOS', fontsize=12)
        else:
            ax=sns.barplot(x=null_percentage.index, y=null_percentage.values,color=random_color() )
            axes.set_xlabel(f'Campos de " {name} "', fontsize=12)
            axes.set_ylabel('Porcentaje NULOS', fontsize=12)
            
        # Ajustar los límites del eje Y para dar más espacio
        ax.set_ylim(0, null_percentage.max() +30)  # Deja un 5% adicional en la parte superior
        # Agregar los porcentajes encima de cada barra
        for i, v in enumerate(null_percentage.values):
            if (v!=0):
                ax.text(i, v + 3, f"{v:.4f}%", ha="center", fontsize=10, rotation=90)
        # Rotar etiquetas del eje x sin warnings
        ax.tick_params(axis='x', rotation=90)  # Rotación de 90º
        # Añadir un título a cada subplot
        ax.set_title(f"ANALISIS PORCENTAJE NULOS DEL DATAFRAME '{name}'")  
        # Aumentamos contador de graficos pintados
        nGraficosPintados+=1
        if pintarTraza:
            print(f"nGraficosPintados : {nGraficosPintados}")
        # Aumentamos el contador de columnas
        grafico_columna+=1
    
    if (tituloGeneral!=""):
        # Agregar un título general
        fig.suptitle(tituloGeneral, fontsize=22)
        # Ajustar el espacio para que las etiquetas no se corten
        plt.tight_layout(pad=4)  
    else:
        # Ajustar el espacio para que las etiquetas no se corten
        plt.tight_layout()  
    # Mostrar el gráfico
    plt.show()
    return

def mostrar_campos_nulos(df):
    hayCamposNulos=False
    # Calcular el total de valores nulos y el porcentaje por columna
    null_counts = df.isnull().sum()
    non_null_counts = df.notnull().sum()
    total_counts = len(df)
    null_percentage = (null_counts / total_counts) * 100
    
    # Crear un DataFrame para mostrar los resultados
    null_analysis = pd.DataFrame({
        'Total Nulls': null_counts,
        'Total Non-Nulls': non_null_counts,
        'Total Rows': total_counts,
        'Null Percentage (%)': null_percentage
    })
    df_nulos=null_analysis[null_analysis["Total Nulls"]!=0]
    if len(df_nulos)>0:
        display(df_nulos)
        hayCamposNulos=True
    else:
        print(f"No hay campos NULOS")
    return hayCamposNulos   

def pintar_dataframe_por_bloques(df,columnas_por_bloque=15,nlineas=5):
    NumeroBloques=0
    # Número de columnas por bloque
    #columnas_por_bloque = 15
    
    # Mostrar columnas de 10 en 10
    for i in range(0, len(df.columns), columnas_por_bloque):
        print(f"Mostrando columnas {i + 1} a {i + columnas_por_bloque}")
        display(df.iloc[:, i:i + columnas_por_bloque].head(nlineas))  # Selección por rango de columnas
        print("\n")
        NumeroBloques+=1
    return NumeroBloques

def visualizar_porcentajes_nulos(df,campos,decimales=2):
    for NombreCampoNulos in campos:
        null_count = df[NombreCampoNulos].isna().sum()
        # Contar valores no nulos en la columna 'country'
        not_null_count = df[NombreCampoNulos].notna().sum()
        print(f"Campo: '{NombreCampoNulos}': ")
        print(f"    Valores nulos en '{NombreCampoNulos}': {null_count}")
        print(f"    Valores no nulos en '{NombreCampoNulos}': {not_null_count}")
        print(f"    % NULOS nulos en '{NombreCampoNulos}': {(null_count/(not_null_count+null_count))*100:.{decimales}f}%")
    return


def pintar_columnas_html(columnas,desc_columnas=None,titulo="",FormatoTabla=True):
    # Calcular la longitud máxima de los nombres de columna
    longitud_columna_maxima = max(len(col) for col in columnas)
    #longitud_maxima = max(len(campo) for diccionario in desc_columnas for campo in diccionario.keys())

    # for i, col in enumerate(columnas_categoricas, 1):
    #     print(f"  {i}. {col}")

    
    # Crear una lista HTML
    if titulo=="":
        titulo = "Columnas"
    html = f"<h3>{titulo}:</h3>"
    if not FormatoTabla:
        html += "<ul>"
        for col in columnas:
            if desc_columnas is None:
                html += f"<li><b>{col}</b>:</li> "
            else:
                # Buscar el valor de la clave 'hotel'
                #descripcion= [item[col] for item in desc_columnas if col in item]
                descripcion=next((item[col] for item in desc_columnas if col in item), "")
                if len(descripcion)==0:
                    descripcion=""
                #print(f"{col}: {descripcion} -> {len(descripcion)}")
                html += f"<li><b>{col}</b>: {descripcion}</li>"
                #html_lista += f"<li><b>{col}</b>:</li> "
        html += "</ul>"
    else:
        # Crear la tabla HTML
        html += f"""
        <style>
            .custom-table {{
                width: 100%; /* La tabla ocupa el 100% del contenedor */
                border-collapse: collapse; /* Elimina el espacio entre celdas */
            }}
            .custom-table th, .custom-table td {{
                padding: 8px; /* Espaciado interno de las celdas */
                border-bottom: 1px solid #ddd; /* Línea inferior para cada fila */
                text-align: left  !important; /* Alineación del texto a la izquierda */
            }}
            .custom-table th {{
                background-color: #f2f2f2; /* Color de fondo para el encabezado */
                font-weight: bold; /* Texto en negrita para el encabezado */
            }}
            .custom-table td:first-child {{
                width: 150px; /* Ancho fijo para la primera columna (Campo) */
                font-weight: bold; /* Texto en negrita para la columna "Campo" */
            }}
        </style>
        <table class="custom-table">
            <colgroup>
                <col style="width: {longitud_columna_maxima*7}px;"> <!-- Ancho fijo para la primera columna (Campo) -->
                <col style="width: auto;"> <!-- La segunda columna ocupará el espacio restante -->
            </colgroup>            
            <tr>
                <th>CAMPO</th>
                <th>DESCRIPCIÓN</th>
            </tr>
            
        """

        for col in columnas:
            if desc_columnas is None:
                html += f"<li><b>{col}</b>:</li> "
            else:
                descripcion=next((item[col] for item in desc_columnas if col in item), "")
                if len(descripcion)==0:
                    descripcion=""
                # Creamos el TR
                html += f"""
                <tr>
                    <td>{col}:</td>
                    <td>{descripcion}</td>
                </tr>
                """
        html += "</table>"
        
    # Mostrar la lista en HTML
    return HTML(html)

def dame_tablas_pycountry(Tipo):
    lista=[]
    match Tipo:
        case "countries":
            # alpha_2='AW', alpha_3='ABW', flag='🇦🇼', name='Aruba', numeric='533'
            lista = [
               {
                "alpha_2": country.alpha_2,
                "alpha_3": country.alpha_3,
                #"flag": country.flag,
                "name": country.name,
                "numeric": country.numeric
                } 
                for country in pycountry.countries
            ]
        case "subdivisions":
            # {'_fields': {'code': 'AD-02', 'name': 'Canillo', 'type': 'Parish', 'parent_code': None, 'country_code': 'AD'}, 'country_code': 'AD'}
            lista = [
               {
                "country_code": subdivision.country_code,
                "code": subdivision.code,
                "name": subdivision.name,
                "type": subdivision.type,
                "parent_code": subdivision.parent_code
                } 
                for subdivision in pycountry.subdivisions
            ]
        case "languages":
            # 'alpha_3': 'aaa', 'name': 'Ghotuo', 'scope': 'I', 'type': 'L'
            lista = [
               {
                "alpha_3": language.alpha_3,
                "name": language.name,
                "scope": language.scope,
                "type": language.type
                } 
                for language in pycountry.languages
            ]
        case "currencies":
            # 'alpha_3': 'AED', 'name': 'UAE Dirham', 'numeric': '784'
            lista = [
               {
                "alpha_3": currency.alpha_3,
                "name": currency.name,
                "numeric": currency.numeric
                } 
                for currency in pycountry.currencies
            ]
        case _:
            return "Opción no válida"
    # Crear el DataFrame
    if len(lista)>0:
        df = pd.DataFrame(lista)
    else:
        df = pd.DataFrame()
    return df


def CSV(operacion,NombreFichero,df=None,ruta="",carpeta="",VisualizarResultado=True,encoding= "utf-8-sig"): 
    if (ruta==""):
        # Obtener la ruta del directorio actual
        directorio_actual = os.getcwd()  # Directorio actual donde está el notebook
        # Construir la ruta completa al archivo en la carpeta "data"
        ruta_fichero_original = os.path.join(directorio_actual, carpeta, NombreFichero)
        # Reemplazar las barras invertidas por barras normales
        ruta_fichero_modificada = ruta_fichero_original.replace("\\", "/")
        #Crear el directorio "Data" si no existe
        if (carpeta!=""):
            os.makedirs("Data", exist_ok=True)
    else:
        # Reemplazar las barras invertidas por barras normales
        ruta_fichero_modificada = ruta.replace("\\", "/")
    # Analizamos la operacion
    match operacion:
        case "LEER":
            # Cargamos el csv en un dataframe
            df = pd.read_csv(ruta_fichero_modificada, encoding= encoding, low_memory=False)
        case "GRABAR":
            #Guardar el DataFrame como un archivo CSV
            #df.to_csv(f"{carpeta}/{NombreFichero}", index=False, encoding=encoding)  # index=False para no guardar el índice como columna
            df.to_csv(ruta_fichero_modificada, index=False, encoding=encoding, errors='replace')  # index=False para no guardar el índice como columna
        case _:
            print(f"Tipo: '{Tipo.upper()}' no válido!")
            VisualizarResultado=False
            df=None
    if VisualizarResultado:
        # Display basic information
        print(f"Ruta: '{ruta_fichero_modificada.upper()}'")
        print(f"Dataset '{NombreFichero}' cargado OK!")
        filas,columnas=df.shape
        print(f"{filas} Filas  y {columnas} Columnas")
    return df        

def pintar_caracteristicas_columnas(df_original,columnas,NumeroEjemplos=10,ValorMayorQue=0):
    for col in columnas:
        valores_unicos = df_original[col].unique()
        cantidad_valores=len(valores_unicos)
        if (ValorMayorQue==0 or cantidad_valores>=ValorMayorQue):
            print(f"Columna: {col}")
            print(f"Tipo: {df_original[col].dtype}")
            print(f"Ejemplo de valores: {df_original[col].head(NumeroEjemplos).tolist()}")
            print(f"Cantidad de valores: {min(NumeroEjemplos,cantidad_valores)}/{cantidad_valores}")
            if cantidad_valores<=NumeroEjemplos:
                print(f"Ejemplo de valores: {df_original[col].unique()[:cantidad_valores]}")
                # Convertir a un diccionario con valores secuenciales
                diccionario = {valor: indice for indice, valor in enumerate(valores_unicos[:cantidad_valores])}
            else:
                print(f"Ejemplo de valores: {df_original[col].unique()[:NumeroEjemplos]} ...")
                # Convertir a un diccionario con valores secuenciales
                diccionario = {valor: indice for indice, valor in enumerate(valores_unicos[:NumeroEjemplos])}
            print(diccionario)
            print("-" * 90)
    return

# Función para convertir alpha-3 a nombres de país
def dame_nombre_pais(alpha3):
    try:
        return pycountry.countries.get(alpha_3=alpha3).name
    except AttributeError:
        return None

# Función para obtener el continente desde el código alpha-3
def dame_codigo_continente(alpha_3):
   
    try:
        country_code = pycountry.countries.get(alpha_3=alpha_3).alpha_2  # Convertir a código Alpha-2
        continent_code = pc.country_alpha2_to_continent_code(country_code)  # Obtener código de continente
        return continent_code
    except:
        continent_code= "--"
    if  continent_code in ("--"):
        # Estos paises no tienen contiente en tablas asi que los vamos a clasificar manual
        # ATA	Antarctica	Antarctica
        # ATF	French Southern Territories	Antarctica (o Oceania)
        # BVT	Bouvet Island	Antarctica
        # ESH	Western Sahara	Africa
        # HMD	Heard Island and McDonald Islands	Antarctica
        # PCN	Pitcairn	Oceania
        # SXM	Sint Maarten (Dutch part)	North America
        # TLS	Timor-Leste	Asia
        # UMI	United States Minor Outlying Islands	Oceania (Pacific)
        # VAT	Holy See (Vatican City State)	Europe
        match alpha_3:
            # ATA	Antarctica	Antarctica
            case "ATA":
                continent_code="AN"
            # ATF	French Southern Territories	Antarctica (o Oceania)
            case "ATF":
                continent_code="AN"
            # BVT	Bouvet Island	Antarctica
            case "BVT":
                continent_code="AN"
            # ESH	Western Sahara	Africa
            case "ESH":
                continent_code="AF"
            # HMD	Heard Island and McDonald Islands	Antarctica
            case "HMD":
                continent_code="AN"
            # PCN	Pitcairn	Oceania
            case "PCN":
                continent_code="OC"
            # SXM	Sint Maarten (Dutch part)	North America
            case "SXM":
                continent_code="NA"
            # TLS	Timor-Leste	Asia
            case "TLS":
                continent_code="AS"
            # UMI	United States Minor Outlying Islands	Oceania (Pacific)
            case "UMI":
                continent_code="OC"
            # VAT	Holy See (Vatican City State)	Europe
            case "VAT":
                continent_code="EU"
            case _:
                continent_code="--"

    return continent_code

# Función para obtener el continente desde el código alpha-3
def dame_nombre_continente(alpha3):
    try:
        continent_code=dame_codigo_continente(alpha3)
        continent_names = {
            "AF": "África",
            "AS": "Asia",
            "EU": "Europa",
            "NA": "América del Norte",
            "SA": "América del Sur",
            "OC": "Oceanía",
            "AN": "Antártica",
        }
        return continent_names.get(continent_code, "Desconocido")
    except:
        return "Desconocido"

