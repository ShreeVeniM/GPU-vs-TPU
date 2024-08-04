import matplotlib.pyplot as plt
import os

def save_plot(history, filename):
    try:
        os.makedirs('charts', exist_ok=True)
        plt.figure()
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.savefig(f'charts/{filename}', dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error in save_plot: {e}")
