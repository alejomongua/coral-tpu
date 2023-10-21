import os
import argparse

import numpy as np
from PIL import Image
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def evaluate_model(
    folder_path,
    model_path,
    labels_file,
    input_mean=128.0,
    input_std=128.0,
    top_k=1,
    threshold=0.0,
):
    """
    # Uso de la función
    folder_path = "ruta_de_tu_carpeta"
    model_path = "ruta_de_tu_modelo.tflite"
    labels_file = "ruta_de_tu_archivo_de_etiquetas.txt"
    metrics = evaluate_model(folder_path, model_path, labels_file)
    print(metrics)
    """
    interpreter = prepare_model(model_path)
    true_labels = []
    predicted_labels = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            true_label = int(
                filename.split("_")[1].split(".")[0]
            )  # Extraer la etiqueta verdadera del nombre del archivo
            true_labels.append(true_label)

            image_path = os.path.join(folder_path, filename)
            prepared_input = prepare_input(
                image_path, input_mean, input_std, interpreter
            )
            label_and_score = infer_and_get_label(
                interpreter, prepared_input, top_k, threshold, labels_file
            )

            # Asumiendo que solo estás interesado en la etiqueta con la mayor puntuación
            predicted_label = int(label_and_score[0][0])
            predicted_labels.append(predicted_label)

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average="weighted"
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score,
    }


def prepare_model(model_path):
    interpreter = make_interpreter(*model_path.split("@"))
    interpreter.allocate_tensors()
    if common.input_details(interpreter, "dtype") != np.uint8:
        raise ValueError("Only support uint8 input type.")
    return interpreter


def prepare_input(image_path, input_mean, input_std, interpreter):
    image = Image.open(image_path)
    image = np.asarray(image).reshape((28, 28, 1))
    params = common.input_details(interpreter, "quantization_parameters")
    scale = params["scales"]
    zero_point = params["zero_points"]
    mean = input_mean
    std = input_std
    if abs(scale * std - 1) < 1e-5 and abs(mean - zero_point) < 1e-5:
        # Input data does not require preprocessing.
        return image
    else:
        # Input data requires preprocessing
        normalized_input = (np.asarray(image) - mean) / (std * scale) + zero_point
        np.clip(normalized_input, 0, 255, out=normalized_input)
        return normalized_input.astype(np.uint8)


def infer_and_get_label(interpreter, prepared_input, top_k, threshold, labels_file):
    common.set_input(interpreter, prepared_input)
    interpreter.invoke()
    classes = classify.get_classes(interpreter, top_k, threshold)
    labels = read_label_file(labels_file) if labels_file else {}
    return [(labels.get(c.id, c.id), c.score) for c in classes]


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-m", "--model", required=True, help="File path of .tflite file."
    )
    parser.add_argument("-i", "--input", help="Image to be classified.")
    parser.add_argument("-f", "--folder", help="Folder with labeled images.")
    parser.add_argument("-l", "--labels", help="File path of labels file.")
    parser.add_argument(
        "-k",
        "--top_k",
        type=int,
        default=1,
        help="Max number of classification results",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.0,
        help="Classification score threshold",
    )
    parser.add_argument(
        "-c", "--count", type=int, default=5, help="Number of times to run inference"
    )
    parser.add_argument(
        "-a",
        "--input_mean",
        type=float,
        default=128.0,
        help="Mean value for input normalization",
    )
    parser.add_argument(
        "-s",
        "--input_std",
        type=float,
        default=128.0,
        help="STD value for input normalization",
    )
    args = parser.parse_args()

    if args.folder:
        metrics = evaluate_model(
            args.folder,
            args.model,
            args.labels,
        )
        print(metrics)
        return

    interpreter = prepare_model(args.model)
    prepared_input = prepare_input(
        args.input, args.input_mean, args.input_std, interpreter
    )
    label_and_score = infer_and_get_label(
        interpreter, prepared_input, args.top_k, args.threshold, args.labels
    )

    print("-------RESULTS--------")
    for label, score in label_and_score:
        print(f"{label}: {score:.5f}")


if __name__ == "__main__":
    main()
