import tensorflow as tf
from tensorflow.keras import layers, Model


def create_model(input_shape=(416, 416, 3)):
    input_img = tf.keras.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(1024, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)

    # Salida: probabilidad + coordenadas de la bounding box
    output = layers.Dense(5, activation="linear")(x)

    model = Model(inputs=input_img, outputs=output)
    return model


# Crear el modelo
model = create_model()
model.summary()
