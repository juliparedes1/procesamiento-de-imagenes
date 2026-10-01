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

img_vacia_th = img_vacia <= 0
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
"""
img1_th = img1 <= 0
plt.imshow(img1_th, cmap='gray'), plt.title("Imagen binaria")
plt.show()

img1_rows = np.sum(img1_th, axis=0) 
np.unique(img1_rows) # -> elegimos un umbral mayor o igual 400

img1_rows_th = img1_rows >= 400
img1_rows_th
np.argwhere(img1_rows_th)

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

# Prueba para ver el umbral de las columnas sobre la totalidad de las imagenes
diccionario_columnas = {}
for i, img in enumerate([img1, img2, img3, img4]):
    img_th = img <= 0
    img_cols = np.sum(img_th, axis=0)
    print(f"Imagen {i}: {np.unique(img_cols)}")
    # Tenemos el umbral (th >= 548) para saber que columnas de la imagen separan a los campos
    img_cols_th = img_cols >= 548
    print(f"Imagen {i}: {np.argwhere(img_cols_th)}")

# Intento de recortar las columnas de cada imagen 

diccionario_columnas = {}
for i, img in enumerate([img1, img2, img3, img4]):
    img_th = img <= 0
    img_cols = np.sum(img_th, axis=0)
    print(f"Imagen {i}: {np.unique(img_cols)}")
    
    indices_columnas = np.where(img_cols >= 548)[0] # Hay que el primer elemento de la tupla
    print(f"Índices detectados para Imagen {i}: {indices_columnas}")

    # Iteramos sobre los índices encontrados para cortar la imagen
    for col in range(1, 8):  # Itera 7 veces (de 1 a 7)
        index = col - 1  # Ajustamos el índice para empezar en 0
        
        if index + 1 < len(indices_columnas): # Nos aseguramos de no pasarnos del límite de columnas detectadas
            inicio = indices_columnas[index]
            fin = indices_columnas[index + 1]
            
            # Ahora sí recortamos usando posiciones numéricas válidas
            diccionario_columnas[col] = img[:, inicio:fin]
            
            plt.imshow(diccionario_columnas[col], cmap="gray")
            plt.title(f"Imagen {i} - Columna {col}")
            plt.show()

Idea para continuar: seguir en este for pero ahora iterando sobre cada columna, la idea de agarrar la fila es lo mismo con la columna,
se puede copiar el código de prueba que está al principio para agarrar las filas. Con esto hecho ya tendríamos las celdas, habría que verificar
que no quede ninguna linea del borde de la celda dentro de la imagen recortada para no generar ruido, eso se puede hacer
achicando la imagen sumandole el indice de columna y fila en 3 por ejemplo, ya que despues a cada imagen de celda hay que aplicarle
cv2.connectedComponentsWithStats(image, connectivity, ltype) para hacer la verificación de los caracteres que eso habría que pensarlo,
como también hay que pensar como guardar todas las celdas en alguna especie de lista de diccionarios para más adelante.
"""