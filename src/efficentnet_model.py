import numpy as np
import os, shutil, pathlib
import datetime

import optuna

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Conv2D,
    MaxPooling2D,
    GlobalAveragePooling2D,
    BatchNormalization,
    Add,
    Input,
    Activation,
    AveragePooling2D,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard
from keras_efficientnets import EfficientNetB0


def cosine_annealing_lr(epoch, lr, num_epochs=50):
    return lr * 0.5 * (1 + np.cos(np.pi * epoch / num_epochs))


def load_images(base_path, batch_size=512):

    rand_seed = 42

    def preprocessing_function(x):

        return x / 255.0

    image_datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=0.2,
        shear_range=0.05,
        zoom_range=0.05,
        preprocessing_function=preprocessing_function,
        validation_split=0.0,
    )

    image_generator = image_datagen.flow_from_directory(
        base_path + "/train",
        class_mode="categorical",
        seed=rand_seed,
        target_size=(224, 224),
        batch_size=batch_size,
        color_mode="rgb",
    )

    image_generator_val = image_datagen.flow_from_directory(
        base_path + "/val",
        class_mode="categorical",
        seed=rand_seed,
        target_size=(224, 224),
        batch_size=batch_size,
        color_mode="rgb",
    )

    # image_generator_test =image_datagen.flow_from_directory(base_path +'/test',
    #                                                class_mode='categorical',
    #                                                seed=rand_seed,
    #                                                target_size=(64, 64),
    #                                                batch_size=batch_size,
    #                                                color_mode='rgb',
    #                                                subset='validation'
    #                                                )

    return image_generator, image_generator_val


def create_model(num_classes=200, input_shape=(224, 224, 3)):
    model = EfficientNetB0(weights=None, include_top=False, input_shape=input_shape)

    # EfficientNetの最後の全結合層を追加します
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    # モデルを結合します
    model = keras.models.Model(inputs=model.input, outputs=predictions)

    return model


# 定義したモデルをコンパイルし、学習を行う関数を定義
def train_and_evaluate_model(trial, base_path, batch_size=512):
    # Generate the hyperparameter values based on the trial
    learning_rate = trial.suggest_float(
        "learning_rate", 1e-5, 1e-2, log=True
    )  # Log scale for learning rate search

    # Load the images using the provided function
    image_generator, image_generator_val = load_images(base_path, batch_size)

    # Create the model using the provided function
    model = create_model()

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Define the Learning Rate Scheduler callback
    lr_scheduler = LearningRateScheduler(
        lambda epoch, lr: cosine_annealing_lr(epoch, lr, num_epochs=50)
    )

    # Define TensorBoard callback to log metrics
    log_dir = "../logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=10)

    # Train the model
    model.fit(
        image_generator,
        validation_data=image_generator_val,
        epochs=50,
        verbose=1,
        callbacks=[lr_scheduler, tensorboard_callback],
    )

    # Evaluate the model on the test set
    score = model.evaluate(image_generator_val, verbose=0)

    # Save the trained model
    model.save(f"{log_dir}/model_{trial.number}.h5")

    return score[1]  # Return the accuracy


def objective(trial):
    # Set the base path where the data is located
    base_path = "/data00/public/ayumi/classifier/tiny_imagenet/tiny-imagenet-200"

    # Define the batch size for the image generators
    batch_size = 512

    # Call the training and evaluation function with the current trial
    accuracy = train_and_evaluate_model(trial, base_path, batch_size)

    # Log hyperparameters, loss, and accuracy to TensorBoard
    # trial.set_user_attr("num_layer", trial.params['num_layer'])
    # trial.set_user_attr("mid_units", trial.params['mid_units'])
    # trial.set_user_attr("num_filters", trial.params['num_filters'])
    # trial.set_user_attr("dropout_rate", trial.params['dropout_rate'])
    trial.set_user_attr("learning_rate", trial.params["learning_rate"])
    trial.set_user_attr("accuracy", accuracy)

    return accuracy


# Run the optimization
study = optuna.create_study(direction="maximize")  # We want to maximize accuracy
study.optimize(
    objective, n_trials=100
)  # You can set the number of trials as per your requirement
