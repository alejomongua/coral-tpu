import os
import time
import sys
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite


def load_image(file_path):
    # Cargar la imagen
    img = Image.open(file_path)

    # Convertir la imagen a escala de grises si tiene canales de color
    if img.mode != "L":
        img = img.convert("L")

    # Redimensionar la imagen a 28x28 píxeles
    img = img.resize((28, 28))

    # Convertir la imagen a un array de numpy
    img_array = np.asarray(img, dtype=np.uint8)

    # Cambiar la forma del array a la esperada por el modelo de TensorFlow Lite
    # [1, 28, 28, 1] que corresponde a [batch_size, height, width, channels]
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension
    img_array = np.expand_dims(img_array, axis=-1)  # channel dimension

    return img_array


def main(model_path, dirname):
    # Cargar el modelo de TFLite
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Obtener información de entrada y salida
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    times = []

    # Liste el directorio
    for image_path in os.listdir(dirname):
        input_data = load_image(os.path.join(dirname, image_path))

        start_time = time.time()
        # Establecer la función de entrada y ejecutar la inferencia
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()

        # Obtener la predicción de la salida
        tflite_results = interpreter.get_tensor(output_details[0]["index"])
        times.append(time.time() - start_time)

    print(f"Tiempo promedio de inferencia: {round(np.mean(times) * 1000, 2)} ms")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Uso: {sys.argv[0]} <model_path> <dirname>")
        sys.exit(1)

    model_path = sys.argv[1]
    dirname = sys.argv[2]

    main(model_path, dirname)
