from enum import Enum

from deepexplain.tensorflow import DeepExplain
from vis.visualization import visualize_saliency, visualize_cam
from vis.utils import utils as vutils

import tensorflow as tf
import numpy as np

import keras
from keras.models import Model
from keras import backend as K

class ExplanationMethods(Enum):
    SALIENCY = 'saliency'
    INTEGRATED_GRADIENTS = 'intgrad'
    GRADIENT_INPUT = 'grad*input'
    DEEPLIFT = 'deeplift'
    ELRP = 'elrp'
    BACKPROP = 'backprop'
    GRADCAM = 'gradcam'

class BaseExplanation():
    def explain(self):
        raise NotImplementedError

class KerasExplanation(BaseExplanation):
    def __init__(self, methods, xs, ys, target=-1, batch_size=None):
        self.methods = methods
        self.target = target
        self.xs = xs
        self.ys = ys
        self.batch_size = batch_size

    def explain(self, model, **kwargs):
        """
        Returns a list of explanations for each method in methods
        """
        with DeepExplain(session=K.get_session()) as de:
            input_tensor = model.layers[0].input    # get input tensor
            fModel = Model(inputs=input_tensor, outputs = model.layers[self.target].output)
            target_tensor = fModel(input_tensor)    # get output tensor

            result = []
            for i in range(len(self.methods)):
                method = self.methods[i]

                if method is ExplanationMethods.BACKPROP:
                    gbp = []
                    for i in range(len(self.xs)):
                        class_idx = np.nonzero(self.ys[i])[0][0]
                        gbp.append(visualize_saliency(model, -1, filter_indices=class_idx, 
                                                            seed_input=self.xs[i], 
                                                            backprop_modifier='guided'))
                    result.append(np.array(gbp))
                elif method is ExplanationMethods.GRADCAM:
                    gradcam = []
                    for i in range(len(self.xs)):
                        class_idx = np.nonzero(self.ys[i])[0][0]
                        gradcam.append(visualize_cam(model, -1, filter_indices=class_idx, 
                                                            seed_input=self.xs[i], 
                                                            backprop_modifier='guided'))
                    result.append(np.array(gradcam))
                else:
                    explanation = de.explain(method.value, target_tensor, input_tensor, self.xs, ys=self.ys, batch_size=self.batch_size)
                    result.append(explanation)
        return result

    def get_method_names(self):
        return [method.value for method in self.methods]