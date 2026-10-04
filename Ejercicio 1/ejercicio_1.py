from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


try:
    BASE_DIR = Path(__file__).parent
except NameError:
    BASE_DIR = Path.cwd() / "Ejercicio 1" if (Path.cwd() / "Ejercicio 1").exists() else Path.cwd()
IMG_PATH = BASE_DIR / "Imagen_con_detalles_escondidos.tif"


# --- Imagen Original -------------------------------------------------------------------------

img = cv2.imread(str(IMG_PATH), cv2.IMREAD_GRAYSCALE)
img.min()
img.max()
np.unique(img)
img_heq = cv2.equalizeHist(img)     # Ecualización global, para comparar

ax1 = plt.subplot(221)
plt.imshow(img, cmap='gray', vmin=0, vmax=255), plt.title('Imagen Original')
plt.subplot(222)
plt.hist(img.flatten(), 256, range=[0, 256]), plt.title('Histograma')
plt.subplot(223, sharex=ax1, sharey=ax1)
plt.imshow(img_heq, cmap='gray', vmin=0, vmax=255), plt.title('Ecualización global')
plt.subplot(224)
plt.hist(img_heq.flatten(), 256, range=[0, 256]), plt.title('Histograma')
plt.show()



# Ecualización local de histograma 
def ecualizacion_local(img, M, N):
    # img    : Imagen de entrada en escalas de grises (2D), formato uint8.
    # M, N   : Tamaño de la ventana (alto x ancho). Enteros positivos, preferentemente impares
    #          para que la ventana tenga un píxel central.
    # img_eq : Imagen de salida, mismo tamaño y tipo que la de entrada.
    m, n = M // 2, N // 2
    img_pad = cv2.copyMakeBorder(img, m, M - m - 1, n, N - n - 1, cv2.BORDER_REPLICATE)  # Agrego bordes para que la ventana no se salga de la imagen
    img_eq = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ventana = img_pad[i:i+M, j:j+N]             # Ventana centrada en el píxel (i,j) de la imagen original
            ventana_eq = cv2.equalizeHist(ventana)      # Transformación de ecualización calculada con el histograma local
            img_eq[i, j] = ventana_eq[m, n]             # Me quedo sólo con el nuevo valor del píxel central
    return img_eq


# Análisis de la imagen con ecualización local 
img_eloc = ecualizacion_local(img, 15, 15)

plt.figure()
ax1 = plt.subplot(121)
plt.imshow(img, cmap='gray', vmin=0, vmax=255), plt.title('Imagen Original')
plt.subplot(122, sharex=ax1, sharey=ax1)
plt.imshow(img_eloc, cmap='gray', vmin=0, vmax=255), plt.title('Ecualización local - ventana 15x15')
plt.show()


# Influencia del tamaño de la ventana 
ventanas = [(3, 3), (7, 7), (15, 15), (31, 31), (51, 51), (101, 101)]

plt.figure()
for k, (M, N) in enumerate(ventanas):
    img_eloc = ecualizacion_local(img, M, N)
    if k == 0:
        ax1 = plt.subplot(2, 3, k+1)
    else:
        plt.subplot(2, 3, k+1, sharex=ax1, sharey=ax1)
    plt.imshow(img_eloc, cmap='gray', vmin=0, vmax=255)
    plt.title(f'Ventana {M}x{N}')
    plt.xticks([]), plt.yticks([])
plt.show()