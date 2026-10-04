from pathlib import Path
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Leemos las imagenes

img_vacia = cv2.imread('grade_sheet_empty.png', cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread('grade_sheet_1.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('grade_sheet_2.png', cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread('grade_sheet_3.png', cv2.IMREAD_GRAYSCALE)
img4 = cv2.imread('grade_sheet_4.png', cv2.IMREAD_GRAYSCALE)

# Mostramos para que se vean bien

ax1 = plt.subplot(221)
plt.imshow(img1, cmap='gray', vmin=0, vmax=255), plt.title('Primera imagen')
plt.subplot(222, sharex=ax1, sharey=ax1)
plt.imshow(img2, cmap='gray', vmin=0, vmax=255), plt.title('Segunda imagen')
plt.subplot(223, sharex=ax1, sharey=ax1)
plt.imshow(img3, cmap='gray', vmin=0, vmax=255), plt.title('Tercera imagen')
plt.subplot(224, sharex=ax1, sharey=ax1)
plt.imshow(img4, cmap='gray', vmin=0, vmax=255), plt.title('Cuarta imagen')
plt.show()

# Observamos los valores únicos de cada imagen

for i, img in enumerate([img_vacia, img1, img2, img3, img4]):
    print(f"Imagen {i+1}: {np.unique(img)} - Tamaño: {img.shape}")

# Intentamos identificar las celdas en la imagen vacía

# Empezamos viendo las columnas

img_vacia_th = img_vacia == 0
plt.imshow(img_vacia_th, cmap='gray'), plt.title("Imagen binaria")
plt.show()

img_vacia_cols = np.sum(img_vacia_th, axis=0) 
np.unique(img_vacia_cols) # -> elegimos un umbral mayor o igual 400

img_vacia_cols_th = img_vacia_cols >= 400
img_vacia_cols_th
np.argwhere(img_vacia_cols_th)

# Para ver la columna de legajo
plt.imshow(img_vacia[:,149:250], cmap='gray'), plt.title("Imagen binaria")
plt.show()

# Prueba para ver la idea de como realizarlo

img1_th = img1 == 0
ax1 = plt.subplot(121)
plt.imshow(img1, cmap='gray', vmin=0, vmax=255), plt.title('Imagen 1')
plt.subplot(122, sharex=ax1, sharey=ax1)
plt.imshow(img1_th, cmap='gray'), plt.title("Imagen 1 binaria")
plt.show()

img1_cols = np.sum(img1_th, axis=0) 
np.unique(img1_cols) # -> elegimos un umbral mayor o igual 400

img1_cols_th = img1_cols >= 500
img1_cols_th
np.argwhere(img1_cols_th)

# Se toma img1 y se hace el crop sobre la columna legajo para luego dividir por registros 

prueba = img1[:,189:315]
plt.imshow(img1[:,189:315], cmap='gray'), plt.title("Imagen binaria")
plt.show()


prueba_th = prueba <= 0
plt.imshow(prueba_th, cmap='gray'), plt.title("Imagen binaria")
plt.show()

prueba_rows = np.sum(prueba_th, axis=1) 
np.unique(prueba_rows) # -> elegimos un umbral mayor o igual 126

prueba_rows_th = prueba_rows >= 126
prueba_rows_th
np.argwhere(prueba_rows_th)
 
celda_prueba = prueba[210:286,:]
plt.imshow(celda_prueba, cmap='gray'), plt.title("Imagen binaria")
plt.show()

# Punto a

# obtener las celdas de cada columnas de cada imagen 

diccionario_imagenes = {}

for i, img in enumerate([img1,img2,img3,img4]): 
    imagen = f"Imagen: {i+1}"
    diccionario_imagenes[imagen] = {}
    img_th = img == 0
    img_cols = np.sum(img_th, axis=0)
    #print(f"Imagen {i+1}: {np.unique(img_cols)}")
    
    indices_columnas = np.argwhere(img_cols >= np.unique(img_cols)[-2])
    #print(f"Índices detectados para Imagen {i+1}: {indices_columnas}")

    indices_columnas = indices_columnas.flatten() # pasar a una dimension 

    # Iteramos sobre los índices encontrados para cortar la imagen
    for col in range(len(indices_columnas) - 1):  # Itera 7 veces (de 1 a 7)
        columna = f"Columna: {col + 1}"
        diccionario_imagenes[imagen][columna] = {} 
        
        inicio_columna = indices_columnas[col]
        fin_columna = indices_columnas[col + 1]
            
        # Ahora sí recortamos usando posiciones numéricas válidas
        inicio_columna_sin_borde = inicio_columna + 3
        fin_columna_sin_borde = fin_columna - 3
        columna_recortada = img[:, inicio_columna_sin_borde : fin_columna_sin_borde]

        columna_recortada_th = columna_recortada == 0
        columna_recortada_filas = np.sum(columna_recortada_th, axis=1)
        #print(f"Columna {col}: {np.unique(columna_recortada_filas)}")

        indices_filas = np.argwhere(columna_recortada_filas >= np.unique(columna_recortada_filas)[-1])
        #print(f"Índices detectados para las filas: {indices_filas}")

        indices_filas = indices_filas.flatten()
        indices_filas = indices_filas[-21:]

        for fila in range(len(indices_filas) - 1):
            celda =  f"Celda: {fila + 1}"
            inicio_celda = indices_filas[fila]
            fin_celda = indices_filas[fila+1]

            inicio_celda_sin_borde = inicio_celda + 3
            fin_celda_sin_borde = fin_celda - 3
            diccionario_imagenes[imagen][columna][celda] = columna_recortada[inicio_celda_sin_borde : fin_celda_sin_borde, :]

            
            #plt.imshow(diccionario_imagenes[imagen][columna][celda], cmap="gray", vmin=0, vmax=255)
            #plt.title(f"Imagen {i+1} - Columna {col+1} - Celda {fila+1}")
            #plt.show()

# Analizamos las celdas

def analizar_celda(celda_img):
    # Binarizar e invertir: La función necesita fondo negro (0) y letras blancas (>0)
    celda_bin = (celda_img < 128).astype(np.uint8) * 255
    
    # Obtener componentes conectadas
    # connectivity=8 mira diagonales, útil para letras que a veces se tocan apenas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(celda_bin, connectivity=8)
    
    caracteres_validos = []
    
    # Filtrar ruido y guardar las posiciones
    # Empezamos el bucle en 1 para ignorar el fondo (label 0)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Filtramos ruido: si el área es muy chica, no es un caracter válido
        if area > 1: #mayor a 1 para que tome el " - " como figura 
            x = stats[i, cv2.CC_STAT_LEFT]
            w = stats[i, cv2.CC_STAT_WIDTH]
            caracteres_validos.append((x, w))
            
    cantidad_caracteres = len(caracteres_validos)
    cantidad_palabras = 0
    
    # Contar palabras calculando la distancia entre caracteres
    if cantidad_caracteres > 0:
        cantidad_palabras = 1 # Si hay letras, hay al menos 1 palabra
        
        # Ordenamos los caracteres de izquierda a derecha basándonos en su posición X
        caracteres_validos.sort(key=lambda c: c[0])
        
        # Revisamos el espacio entre cada letra y la siguiente
        for i in range(cantidad_caracteres - 1):
            x_actual, w_actual = caracteres_validos[i]
            x_siguiente, _ = caracteres_validos[i+1]
            
            # El espacio en blanco es donde arranca la siguiente letra menos donde terminó la actual
            distancia_blanco = x_siguiente - (x_actual + w_actual)
            
            # UMBRAL DE ESPACIO: Si hay más de X píxeles de distancia, asumimos que es una nueva palabra.
            if distancia_blanco > 8:
                cantidad_palabras += 1
                
    return cantidad_caracteres, cantidad_palabras

# Arrancamos desde la Columna 2. Al no incluir la "Columna 1" (ID), el bucle for de abajo la ignorará automáticamente 
columnas_validar = {
    "Columna: 2": "Legajo", 
    "Columna: 3": "Nombre y Apellido", 
    "Columna: 4": "Parcial 1", 
    "Columna: 5": "Parcial 2", 
    "Columna: 6": "Parcial 3", 
    "Columna: 7": "Condición Final"
}

for imagen, columnas in diccionario_imagenes.items():
    print(f"\n--- Procesando {imagen} ---")
    
    # Obtenemos la lista de celdas (filas) usando cualquier columna como referencia
    nombres_celdas = list(columnas["Columna: 2"].keys()) 
    
    # Iteramos por el índice de la fila para tener el ID numérico
    for fila_index in range(len(nombres_celdas)):
        id_registro = fila_index + 1  # Esto les da el 1, 2, 3... para el print y el CSV
        celda_nombre = nombres_celdas[fila_index]
        
        print(f"\n> Registro {id_registro}:")
        
        # Este bucle solo recorrerá de la Columna 2 a la 7
        for col_id, tipo_campo in columnas_validar.items():
            
            # Extraemos la imagen de la celda actual
            celda_img = columnas[col_id][celda_nombre]
            
            # Analizamos la celda con cv2.connectedComponentsWithStats
            cant_caracteres, cant_palabras = analizar_celda(celda_img)
            estado = "MAL"
            
            if tipo_campo == "Nombre y Apellido":
                if cant_palabras >= 2 and cant_caracteres <= 12:
                    estado = "OK"
                    
            elif tipo_campo == "Legajo":
                # Asumimos que todo el legajo es "una palabra" continua (sin espacios)
                if cant_caracteres == 8 and cant_palabras == 1:
                    estado = "OK"
                    
            elif tipo_campo in ["Parcial 1", "Parcial 2", "Parcial 3"]:
                if cant_caracteres in [1, 2] and cant_palabras == 1:
                    estado = "OK"
                    
            elif tipo_campo == "Condición Final":
                if cant_caracteres == 1 and cant_palabras == 1:
                    estado = "OK"
            
            # Si detecta 0 caracteres es porque la celda está vacía (MAL)
            if cant_caracteres == 0:
                estado = "MAL"
                
            print(f"> {tipo_campo}: {estado}")


# Punto b y c


# Guardamos el resultado de la validación de cada campo 
def validar_campo(tipo_campo, celda_img):
    # tipo_campo : Nombre del campo ("Legajo", "Nombre y Apellido", "Parcial 1", ..., "Condición Final")
    # celda_img  : Imagen de la celda (sin bordes)
    # estado     : "OK" o "MAL" (mismas reglas que el bucle del punto a)
    cant_caracteres, cant_palabras = analizar_celda(celda_img)
    estado = "MAL"
    if tipo_campo == "Nombre y Apellido":
        if cant_palabras >= 2 and cant_caracteres <= 12:
            estado = "OK"
    elif tipo_campo == "Legajo":
        if cant_caracteres == 8 and cant_palabras == 1:
            estado = "OK"
    elif tipo_campo in ["Parcial 1", "Parcial 2", "Parcial 3"]:
        if cant_caracteres in [1, 2] and cant_palabras == 1:
            estado = "OK"
    elif tipo_campo == "Condición Final":
        if cant_caracteres == 1 and cant_palabras == 1:
            estado = "OK"
    if cant_caracteres == 0:
        estado = "MAL"
    return estado

# Guardamos cada registro en un diccionario para el punto c

resultados = {}   
for imagen, columnas in diccionario_imagenes.items():
    resultados[imagen] = []
    nombres_celdas = list(columnas["Columna: 2"].keys())
    for fila_index in range(len(nombres_celdas)):
        registro = {"ID": fila_index + 1}
        for col_id, tipo_campo in columnas_validar.items():
            registro[tipo_campo] = validar_campo(tipo_campo, columnas[col_id][nombres_celdas[fila_index]])
        resultados[imagen].append(registro)


# --- Punto b: Identificamos la letra de la Condición Final -----------------------------------
def clasificar_condicion(celda_img):
    # celda_img : Imagen de la celda "Condición Final" (con un único caracter, ya validada)
    # Devuelve  : "L", "R" u "Otro"
    # Se diferencian por la forma de la letra:
    #   * L --> No tiene agujeros, tiene palo vertical a la izquierda y nada arriba a la derecha.
    #   * R --> Tiene 1 agujero y palo vertical a la izquierda.
    #   * A --> Tiene 1 agujero pero NO tiene palo vertical a la izquierda (lados inclinados).
    celda_bin = (celda_img < 128).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(celda_bin, connectivity=8)
    idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]) # La componente más grande es la letra
    x, y, w, h = stats[idx, :4]
    letra = (labels[y:y+h, x:x+w] == idx).astype(np.uint8)              # Recorte de la letra (1 = letra, 0 = fondo)

    # Cantidad de agujeros: componentes del fondo que no tocan el borde.
    # Agrego un borde de fondo para que todo el fondo exterior quede conectado en una sola componente.
    fondo = cv2.copyMakeBorder(1 - letra, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
    num_fondo, _ = cv2.connectedComponents(fondo, connectivity=4)
    agujeros = num_fondo - 2                                            # Restamos la etiqueta 0 (la letra) y el fondo exterior

    # Palo vertical: proporción de filas que tienen píxeles en las 2 primeras columnas
    palo_izquierdo = letra[:, :2].any(axis=1).mean() > 0.8
    arriba_derecha = letra[:2, -2:].any()

    if agujeros == 0 and palo_izquierdo and not arriba_derecha:
        return "L"
    if agujeros == 1 and palo_izquierdo:
        return "R"
    return "Otro"


def recortar_contenido(celda_img):
    # Devuelve el "crop" del contenido escrito en la celda (sin el espacio en blanco alrededor)
    celda_bin = (celda_img < 128).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(celda_bin, connectivity=8)
    stats = stats[1:]                                                   # Saco el fondo
    stats = stats[stats[:, cv2.CC_STAT_AREA] > 1]                     # Mismo filtro de ruido que analizar_celda
    x0 = stats[:, cv2.CC_STAT_LEFT].min()
    y0 = stats[:, cv2.CC_STAT_TOP].min()
    x1 = (stats[:, cv2.CC_STAT_LEFT] + stats[:, cv2.CC_STAT_WIDTH]).max()
    y1 = (stats[:, cv2.CC_STAT_TOP] + stats[:, cv2.CC_STAT_HEIGHT]).max()
    return celda_img[y0:y1, x0:x1]


def generar_imagen_desaprobados(desaprobados, titulo):
    # desaprobados : Lista de tuplas (condicion, crop_nombre), condicion = "L" o "R"
    # titulo       : Texto del encabezado
    # img_salida   : Imagen BGR con un renglón por alumno: indicador de color + crop del nombre
    alto_fila, alto_nombre, ancho = 40, 24, 480
    colores = {"R": (0, 140, 255), "L": (0, 0, 220)}                    # BGR: naranja = Recupera, rojo = Libre
    textos = {"R": "RECUPERA", "L": "LIBRE"}

    filas = max(len(desaprobados), 1) + 1                               # +1 por el encabezado
    img_salida = np.full((alto_fila * filas, ancho, 3), 255, dtype=np.uint8)
    cv2.putText(img_salida, titulo, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    if len(desaprobados) == 0:
        cv2.putText(img_salida, "Sin alumnos desaprobados", (10, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    for k, (condicion, crop) in enumerate(desaprobados):
        y = alto_fila * (k + 1)
        # Indicador: rectángulo de color con el texto de la condición
        cv2.rectangle(img_salida, (10, y + 5), (120, y + alto_fila - 5), colores[condicion], -1)
        cv2.putText(img_salida, textos[condicion], (16, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # Crop del nombre: lo escalo a una altura fija (las planillas tienen distintos tamaños de letra)
        escala = alto_nombre / crop.shape[0]
        crop_esc = cv2.resize(crop, (int(crop.shape[1] * escala), alto_nombre), interpolation=cv2.INTER_NEAREST)
        crop_esc = crop_esc[:, :ancho - 140]                            # Por si el nombre no entra
        y0 = y + (alto_fila - alto_nombre) // 2
        img_salida[y0:y0 + alto_nombre, 135:135 + crop_esc.shape[1]] = cv2.cvtColor(crop_esc, cv2.COLOR_GRAY2BGR)
    return img_salida


#  Punto c: CSV con los resultados de la validación 
def guardar_csv(registros, ruta):
    # registros : Lista de diccionarios {"ID": 1, "Legajo": "OK", ...}
    # ruta      : Ruta del archivo CSV de salida
    columnas_csv = ["ID", "Legajo", "Nombre y Apellido", "Parcial 1", "Parcial 2", "Parcial 3", "Condición Final"]
    with open(ruta, "w", newline="", encoding="utf-8-sig") as f:       # utf-8-sig para que Excel muestre bien la "ó"
        writer = csv.DictWriter(f, fieldnames=columnas_csv)
        writer.writeheader()
        writer.writerows(registros)


# --- Punto d: Aplicamos b y c sobre las 4 planillas ------------------------------------------
carpeta_salida = Path("resultados")
carpeta_salida.mkdir(exist_ok=True)

for imagen, registros in resultados.items():
    columnas = diccionario_imagenes[imagen]
    nombres_celdas = list(columnas["Columna: 2"].keys())

    # b) Alumnos desaprobados (sólo registros con todos los campos OK)
    desaprobados = []
    for registro in registros:
        campos = [registro[c] for c in columnas_validar.values()]
        if all(estado == "OK" for estado in campos):
            celda_nombre = nombres_celdas[registro["ID"] - 1]
            condicion = clasificar_condicion(columnas["Columna: 7"][celda_nombre])
            if condicion in ["L", "R"]:
                crop_nombre = recortar_contenido(columnas["Columna: 3"][celda_nombre])
                desaprobados.append((condicion, crop_nombre))
                print(f"{imagen} - Registro {registro['ID']}: Condición {condicion}")

    nombre_archivo = imagen.replace(" ", "_")
    img_desaprobados = generar_imagen_desaprobados(desaprobados, f"Planilla {imagen.split()[-1]} - Alumnos no aprobados")
    cv2.imwrite(str(carpeta_salida / f"desaprobados_{nombre_archivo}.png"), img_desaprobados)

    plt.figure()
    plt.imshow(cv2.cvtColor(img_desaprobados, cv2.COLOR_BGR2RGB)), plt.title(f"Desaprobados - {imagen}")
    plt.xticks([]), plt.yticks([])
    plt.show()

    # c) CSV
    guardar_csv(registros, carpeta_salida / f"validacion_{nombre_archivo}.csv")

print(f"\nResultados guardados en la carpeta: {carpeta_salida.resolve()}")