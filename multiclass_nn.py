import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import SparseCategoricalCrossentropy


def img_recognition_mdl(X_train, y_train, n_epochs: int):
    # We are working with 28 x 28 pixel images, so we flatten the images to shape (784,)
    X_train = X_train.reshape((-1, 784))

    # Creating the NN to solve the Multiclass Classification problem
    model = Sequential([
        keras.Input(shape=(784,)),
        Dense(25, activation='relu'),
        Dense(15, activation='relu'),
        Dense(10, activation='linear')
    ])

    model.summary()
    for i, layer in enumerate(model.layers):
        print(f"Layer {i + 1} - Type: {layer.__class__.__name__}, "
              f"Units: {layer.units}, Activation: {layer.activation.__name__}")

    print('\nAssigning the loss function...')
    # Compiling the model to specify the loss function
    model.compile(
        loss=SparseCategoricalCrossentropy(from_logits=True),  # Reducing Numerical Round-off Errors
        optimizer=keras.optimizers.Adam(learning_rate=1e-3)
    )
    print('SparseCategoricalCrossentropy loss function assigned. \nNow, fitting the model...')
    # Fitting
    model.fit(X_train, y_train, epochs=n_epochs)
    print('Fitting completed.')

    return model

def accuracy(y_true, y_pred):
    """

    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: accuracy score
    """
    correct_preds = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            correct_preds += 1
    return correct_preds / len(y_true)

def img_recognition_test(model: Sequential, X_test, y_test):
    # We are working with 28 x 28 pixel images, so we flatten the images to shape (784,)
    m = X_test.shape[0]
    y_hat = np.zeros_like(y_test)

    print('Model evaluation...')
    for i in range(m):
        pred_i = model.predict(X_test[i].reshape(1, 784))
        pred_prob_i = tf.nn.softmax(pred_i)
        y_hat[i] = np.argmax(pred_prob_i)

    return y_hat, accuracy(y_test, y_hat)
