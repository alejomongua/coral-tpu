import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np


def load_mnist_data():
    (train_images, _), (_, _) = mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1)).astype("float32") / 255
    return train_images


def representative_dataset_gen():
    data = load_mnist_data()
    for image in data[
        :100
    ]:  # Utiliza los primeros 100 ejemplos como un conjunto de datos representativo
        image = np.expand_dims(image, axis=0)
        yield [image]


def quantize_full_integer(model_path, quantized_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8  # o tf.uint8 para TF 2.3 y anteriores
    converter.inference_output_type = tf.uint8  # o tf.uint8 para TF 2.3 y anteriores
    tflite_quant_model = converter.convert()

    # Si quantized_path no termina en .tflite, póngalo
    if not quantized_path.endswith(".tflite"):
        quantized_path += ".tflite"

    # Guardar el modelo cuantizado
    with open(quantized_path, "wb") as f:
        f.write(tflite_quant_model)


def quantize_model(model_path, quantized_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()

    # Si quantized_path no termina en .tflite, póngalo
    if not quantized_path.endswith(".tflite"):
        quantized_path += ".tflite"

    # Guardar el modelo cuantizado
    with open(quantized_path, "wb") as f:
        f.write(tflite_quant_model)


def convert(model, output_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Guardar el modelo TFLite
    with open(output_path, "wb") as f:
        f.write(tflite_model)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python convert.py <path/to/input/model> <path/to/output/model>")
        sys.exit(1)

    path_to_model = sys.argv[1]
    path_to_output_model = sys.argv[2]
    model = tf.keras.models.load_model(path_to_model)
    # convert(model, path_to_output_model)
    # quantize_model(model, path_to_output_model)
    quantize_full_integer(model, path_to_output_model)
