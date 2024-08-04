import tensorflow as tf
import logging
from src.train import train_model_on_gpu, train_model_on_tpu, train_model_on_cpu
from src.utils import save_plot

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info('Starting main function')
    try:
        # Check for GPUs
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            logging.info(f"GPU available: {gpu}")

        # Train model on GPU
        logging.info('Training model on GPU')
        gpu_model, gpu_history = train_model_on_gpu()
        if gpu_history:
            save_plot(gpu_history, 'gpu_training_accuracy.png')

        # Train model on TPU
        logging.info('Training model on TPU')
        tpu_model, tpu_history = train_model_on_tpu()
        if tpu_history:
            save_plot(tpu_history, 'tpu_training_accuracy.png')

        # Train model on CPU
        logging.info('Training model on CPU')
        cpu_model, cpu_history = train_model_on_cpu()
        if cpu_history:
            save_plot(cpu_history, 'cpu_training_accuracy.png')

    except Exception as e:
        logging.error(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
