import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


class MNISTClassifier:
    def __init__(self):
        self.model = self.build_model()
        self.compile_model()

    def build_model(self):
        model = models.Sequential(
            [
                layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.Flatten(),
                layers.Dense(64, activation="relu"),
                layers.Dense(10, activation="softmax"),
            ]
        )
        return model

    def compile_model(self):
        self.model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

    def train_model(
        self, train_images, train_labels, epochs=5, batch_size=64, validation_split=0.1
    ):
        history = self.model.fit(
            train_images,
            train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
        )
        return history

    def evaluate_model(self, test_images, test_labels):
        test_loss, test_acc = self.model.evaluate(test_images, test_labels)
        return test_loss, test_acc

    def save_model(self, filepath):
        # Guardar el modelo en formato TensorFlow SavedModel
        self.model.save(filepath)


def load_and_preprocess_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1)).astype("float32") / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype("float32") / 255
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    return (train_images, train_labels), (test_images, test_labels)


def main(path):
    (train_images, train_labels), (
        test_images,
        test_labels,
    ) = load_and_preprocess_data()
    classifier = MNISTClassifier()
    classifier.train_model(train_images, train_labels)
    test_loss, test_acc = classifier.evaluate_model(test_images, test_labels)
    print(f"Test accuracy: {test_acc}")
    classifier.save_model(path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python conv_classifier.py <path/to/model>")
        sys.exit(1)

    path_to_model = sys.argv[1]

    main(path_to_model)
