import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Layer

class KerasLayer(Layer):
    def __init__(self, **kwargs):
        super(KerasLayer, self).__init__(**kwargs)
        self.layer = None

    def build(self, input_shape):
        self.layer = hub.KerasLayer("Plant Disease Detection.h5")  # Update the path accordingly
        self.layer.build(input_shape)

    def call(self, inputs):
        return self.layer(inputs)
