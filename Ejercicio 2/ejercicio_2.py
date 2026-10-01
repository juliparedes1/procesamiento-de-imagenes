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

for i, img in enumerate([img1,img2,img3,img4]):
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
        columna = f"Colomuna: {col + 1}"
        diccionario_imagenes[imagen][columna] = {} 
        
        inicio_columna = indices_columnas[col]
        fin_columna = indices_columnas[col + 1]
            
            # Ahora sí recortamos usando posiciones numéricas válidas
        inicio_columna_sin_borde = inicio_columna + 3
        fin_columna_sin_borde = fin_columna - 3
        columna_recortada = img[:, inicio_columna_sin_borde : fin_columna]

        columna_recortada_th = columna_recortada == 0
        columna_recortada_filas = np.sum(columna_recortada_th, axis=1)
        print(f"Columna {col}: {np.unique(columna_recortada_filas)}")

        indices_filas = np.argwhere(columna_recortada_filas >= np.unique(columna_recortada_filas)[-1])
        print(f"Índices detectados para las filas: {indices_filas}")

        indices_filas = indices_filas.flatten()

        for fila in range(len(indices_filas) - 1):
            celda =  f"Celda: {fila + 1}"
            inicio_celda = indices_filas[fila]
            fin_celda = indices_filas[fila+1]

            inicio_celda_sin_borde = inicio_celda + 3
            fin_celda_sin_borde = fin_celda - 3
            diccionario_imagenes[imagen][columna][celda] = columna_recortada[inicio_celda_sin_borde : fin_celda, :]

            
            #plt.imshow(diccionario_imagenes[imagen][columna][celda], cmap="gray", vmin=0, vmax=255)
            #plt.title(f"Imagen {i+1} - Columna {col+1} - Celda {fila+1}")
            #plt.show()
