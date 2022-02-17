from os import system
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


IN_COLAB = False

try:
    import google.colab
    from google.colab import drive
    drive._mount('/content/drive')
    IN_COLAB = True
except:
    IN_COLAB = False

    