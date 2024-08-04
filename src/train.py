import tensorflow as tf
import time
from src.model import get_model
from src.data_loader import get_data

def train_model_on_gpu():
    try:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = get_model()

        x_train, y_train, x_test, y_test = get_data()
        if x_train is None:
            return None, None

        start_time = time.time()

        history = model.fit(x_train,
                            y_train,
                            batch_size=1024,
                            epochs=10,
                            validation_data=(x_test, y_test))

        end_time = time.time()
        processing_time = end_time - start_time
        print(f"GPU Processing time: {processing_time} seconds")
        return model, history
    except Exception as e:
        print(f"Error in train_model_on_gpu: {e}")
        return None, None

def train_model_on_tpu():
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)

        with strategy.scope():
            model = get_model()

        x_train, y_train, x_test, y_test = get_data()
        if x_train is None:
            return None, None

        start_time = time.time()

        history = model.fit(x_train,
                            y_train,
                            batch_size=1024,
                            epochs=10,
                            validation_data=(x_test, y_test))

        end_time = time.time()
        processing_time = end_time - start_time
        print(f"TPU Processing time: {processing_time} seconds")
        return model, history
    except Exception as e:
        print(f"Error in train_model_on_tpu: {e}")
        return None, None

def train_model_on_cpu():
    try:
        model = get_model()

        x_train, y_train, x_test, y_test = get_data()
        if x_train is None:
            return None, None

        start_time = time.time()

        history = model.fit(x_train,
                            y_train,
                            batch_size=1024,
                            epochs=10,
                            validation_data=(x_test, y_test))

        end_time = time.time()
        processing_time = end_time - start_time
        print(f"CPU Processing time: {processing_time} seconds")
        return model, history
    except Exception as e:
        print(f"Error in train_model_on_cpu: {e}")
        return None, None
