import numpy as np
import tensorflow as tf
from concise.preprocessing.sequence import encodeDNA
import keras

def encode_dna(seq):
    return encodeDNA(seq)

def decode_dna(seq):
    return 

def count_dna(seq, allowed_bases=['A','T','G','C']):
    seq = seq.upper()
    total_dna_bases = 0
    for base in allowed_bases:
        total_dna_bases = total_dna_bases + seq.count(base.upper())
    dna_fraction = total_dna_bases / len(seq)
    return(dna_fraction * 100)

def copy_model(model):
    import tensorflow as tf
    sess = tf.InteractiveSession()
    copy = keras.models.clone_model(model)
    copy.set_weights(model.get_weights())
    return copy 

def clone_keras_model(model):
    clone = keras.models.clone_model(model)
    clone.set_weights(model.get_weights())
    return clone

def save_weights(model, weights):
    model.set_weights(weights)

def randomize_layer_weights(model, layer): 
    random_weights = np.random.random(layer.weights.shape)
    print(random_weights.shape)
    print(layer.shape)
    model.layers[layer].set_weights(random_weights)

def get_random_and_current_weights(model):
    # sess = tf.InteractiveSession()
    # tf.initializers.global_variables().run()
    new_weights = [layer.get_weights() for layer in model.layers]
    original_weights = new_weights.copy()
    for l in range(len(new_weights)):
        layer = new_weights[l]
        if len(layer) > 0:
            for i in range(len(layer)):
                var_shape = layer[i].shape
                new_weights[l][i] = np.random.random(var_shape)
    return new_weights, original_weights