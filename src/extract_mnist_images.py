from tensorflow.keras.datasets import mnist
import numpy as np
import os
from PIL import Image


def save_random_images(n, save_dir):
    """
    Uso de la función
    n = 10  # Número de imágenes que deseas extraer
    save_dir = 'ruta_donde_guardar_imagenes'  # Directorio donde se guardarán las imágenes
    save_random_images(n, save_dir)
    """
    (_, _), (test_images, test_labels) = mnist.load_data()
    indices = np.random.choice(
        len(test_images), n, replace=False
    )  # Obtener n índices aleatorios

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # Crear el directorio si no existe

    for i, idx in enumerate(indices, start=1):
        image = test_images[idx]
        label = test_labels[idx]
        filename = f"{str(i).zfill(5)}_{label}.png"  # Formato del nombre del archivo
        filepath = os.path.join(save_dir, filename)  # Ruta completa del archivo
        Image.fromarray(image).save(filepath)  # Guardar la imagen


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <n> <save_dir>")
        sys.exit(1)

    n = int(sys.argv[1])
    save_dir = sys.argv[2]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_random_images(n, save_dir)
