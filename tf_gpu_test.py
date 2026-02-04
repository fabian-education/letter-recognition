import tensorflow as tf

#tensorflow gpu available?
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU(s) gefunden:", gpus)
else:
    print("Keine GPU gefunden. TensorFlow verwendet die CPU.")

#if this test is successful, tensorflow will automatically use
#a single GPU for all functions that are GPU-supported (matmul for example)
#all other functions are executed on the CPU