from pathlib import Path

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
plt.imshow(img1_th, cmap='gray'), plt.title("Imagen binaria")
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


# obtener las celdas de cada columnas de cada imagen 

diccionario_imagenes = {}

for i, img in enumerate([img1]): #img2,img3,img4
    imagen = f"imagen {i+1}"
    diccionario_imagenes[imagen] = {}
    img_th = img == 0
    img_cols = np.sum(img_th, axis=0)
    print(f"Imagen {i+1}: {np.unique(img_cols)}")
    
    indices_columnas = np.argwhere(img_cols >= np.unique(img_cols)[-2])
    print(f"Índices detectados para Imagen {i+1}: {indices_columnas}")

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
        print(f"Columna {col}: {np.unique(columna_recortada_filas)}")

        indices_filas = np.argwhere(columna_recortada_filas >= np.unique(columna_recortada_filas)[-1])
        print(f"Índices detectados para las filas: {indices_filas}")

        indices_filas = indices_filas.flatten()
        indices_filas = indices_filas[-21:]

        for fila in range(len(indices_filas) - 1):
            celda =  f"Celda: {fila + 1}"
            inicio_celda = indices_filas[fila]
            fin_celda = indices_filas[fila+1]

            inicio_celda_sin_borde = inicio_celda + 3
            fin_celda_sin_borde = fin_celda - 3
            diccionario_imagenes[imagen][columna][celda] = columna_recortada[inicio_celda_sin_borde : fin_celda_sin_borde, :]

            
            plt.imshow(diccionario_imagenes[imagen][columna][celda], cmap="gray", vmin=0, vmax=255)
            plt.title(f"Imagen {i+1} - Columna {col+1} - Celda {fila+1}")
            plt.show()



# Analizamos las celdas

def analizar_celda(celda_img):
    # 1. Binarizar e invertir: La función necesita fondo negro (0) y letras blancas (>0)
    celda_bin = (celda_img < 128).astype(np.uint8) * 255
    
    # 2. Obtener componentes conectadas
    # connectivity=8 mira diagonales, útil para letras que a veces se tocan apenas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(celda_bin, connectivity=8)
    
    caracteres_validos = []
    
    # 3. Filtrar ruido y guardar las posiciones
    # Empezamos el bucle en 1 para ignorar el fondo (label 0)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Filtramos ruido: si el área es muy chica, no es un caracter válido
        if area > 3: #mayor a 3 para que tome el " - " como figura 
            x = stats[i, cv2.CC_STAT_LEFT]
            w = stats[i, cv2.CC_STAT_WIDTH]
            caracteres_validos.append((x, w))
            
    cantidad_caracteres = len(caracteres_validos)
    cantidad_palabras = 0
    
    # 4. Contar palabras calculando la distancia entre caracteres
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
            # (Este valor de "8" puede que necesites ajustarlo a 5 o 10 dependiendo de la imagen)
            if distancia_blanco > 8:
                cantidad_palabras += 1
                
    return cantidad_caracteres, cantidad_palabras

# Mapeo corregido: Arrancamos desde la Columna 2.
# Al no incluir la "Columna 1", el bucle for de abajo la ignorará automáticamente 
# a la hora de validar, ahorrando tiempo de procesamiento.
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