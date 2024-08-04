import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical

def get_data():
    try:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        x_train = np.expand_dims(x_train, 3)
        x_test = np.expand_dims(x_test, 3)

        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        return x_train, y_train, x_test, y_test
    except Exception as e:
        print(f"Error in get_data: {e}")
        return None, None, None, None
