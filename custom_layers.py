import tensorflow_hub as hub
from tensorflow.keras.layers import Layer

class KerasLayer(Layer):
    def __init__(self, **kwargs):
        super(KerasLayer, self).__init__(**kwargs)
        self.layer = None  # This will hold the KerasLayer from TFHub

    def build(self, input_shape):
        # Here, replace the handle URL with the one you want to use
        # For example, MobileNetV2's feature vector model
        self.layer = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4")
        
        # Now the layer is built with the input shape provided by the model
        self.layer.build(input_shape)

    def call(self, inputs):
        # Forward pass, passing the input to the loaded layer
        return self.layer(inputs)
