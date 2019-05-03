import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from enum import Enum
import logging
import utils
import explanation
import similarity

# Keras sanity checker:
import keras
import keras.backend as K
from keras.initializers import glorot_uniform  # Or your initializer of choice

# Torch Sanity Checker:
import torch

# Libraries
import numpy as np
import pandas as pd
import argparse

class SanityCheckMethods(Enum):
    COMPLETE_RANDOMIZATION = 'random'
    CASCADING_RANDOMIZATION = 'cascading randomization'
    DATA_RANDOMIZATION = 'data randomization'

class BaseSanityChecker(ABC):
    """
    Takes care of model randomization, running the explanation method, and comparing/outputting results.
    """
    @abstractmethod
    def evaluate(self, **kwargs):
        pass

class KerasSanityChecker(BaseSanityChecker):
    def __init__(self, keras_model):    
        if not isinstance(keras_model, keras.Model):
            raise Exception('Cannot create sanity checker, %s not instance of keras.models.Model', type(keras_model))
        self.model = utils.clone_keras_model(keras_model)

    def evaluate(self, randomization, explanation, similarity_metrics, **kwargs):
        """
        randomization: string name of the randomization to perform
        explanation: the explanation(s) objects to run 
        similarity metrics: list of similarity functions to compute 
        """
        if randomization is SanityCheckMethods.COMPLETE_RANDOMIZATION:
            result = self._check_random(explanation, similarity_metrics) 
        elif randomization is SanityCheckMethods.CASCADING_RANDOMIZATION:
            result = self._check_cascade_random(explanation, similarity_metrics) 
        elif randomization is SanityCheckMethods.DATA_RANDOMIZATION:
            if 'random_data_model' in kwargs:
                random_data_model = kwargs['random_data_model'] 
                result = self._check_data_random(explanation, similarity_metrics, random_data_model)
            else: 
                raise Exception("Data randomization requires random model input \'random_data_model\'")
        else:
            raise Exception("Randomization method not supported.")
        
        logging.info('result shape %s', result.shape)
        dfs = self._generate_dataframes(result, explanation, similarity_metrics)
        # Plot graphs for each explanation method
        if 'plot' in kwargs and kwargs['plot'] == True:
            for i in range(len(dfs)):
                df = dfs[i]
                print(df)
                if len(df.columns) > 1: 
                    fig = df.plot(kind='line')
                    fig.set_ylabel(similarity_metrics[i].__name__)
                    fig.set_xlabel('layers')
                else:
                    fig = df.plot(kind='bar', y=0)
                    fig.set_ylabel(similarity_metrics[i].__name__)
        
        return dfs
    
    def _check_random(self, explanation, similarity_metrics):
        model_copy = utils.clone_keras_model(self.model)
        original_explanation = explanation.explain(model_copy)
        self._randomize_all_layer_weights(model_copy)
        new_explanation = explanation.explain(model_copy)
        return self._get_result(original_explanation, new_explanation, similarity_metrics)

    def _check_cascade_random(self, explanation, similarity_metrics):
        results = np.zeros((len(self.model.layers),                 # placeholder for results
            len(explanation.methods), len(similarity_metrics))) 
        model_copy = utils.clone_keras_model(self.model)            # don't modify the original model
        original_explanation = explanation.explain(model_copy)      # reference the unchanged explanation
        for i in reversed(range(len(model_copy.get_weights()))):    # process layers
            self._randomize_layer_weights(model_copy, i)                # randomize layer
            new_explanation = explanation.explain(model_copy)           # generate explanation
            result = self._get_result(original_explanation, new_explanation, similarity_metrics)
            results[i] = result                                         # aggregate results
        return results

    def _check_data_random(self, explanation, similarity_metrics, random_data_model):
        original_explanation = explanation.explain(self.model)
        new_explanation = explanation.explain(random_data_model)
        return self._get_result(original_explanation, new_explanation, similarity_metrics)
    
    def _generate_dataframes(self, results, explanation, similarity_metrics):
        """
        Returns a list of pandas dataframes for easy plotting
        (layer vs explanation) vs similarity
        """
        dfs = []
        methods = explanation.get_method_names()
        if results.ndim == 3:
            for s in range(len(similarity_metrics)):
                df = pd.DataFrame(
                    data=np.flip(results[:,:,s], axis=0),
                    columns=explanation.get_method_names())
                dfs.append(df)
        elif results.ndim == 2:
            for s in range(len(similarity_metrics)):
                df = pd.DataFrame(data=results[:,s],index=explanation.get_method_names())
                dfs.append(df)
        return dfs

    def _get_result(self, original_explanations, new_explanations, similarity_metrics):
        """
        Return result ( # explanations, # similarity_metrics )
        """
        result = np.zeros((len(original_explanations), len(similarity_metrics)))
        for i in range(len(original_explanations)):
            for j in range(len(similarity_metrics)):
                similarity_metric = similarity_metrics[j]
                similarity = similarity_metric(original_explanations[i], new_explanations[i])
                # print(similarity_metric.__name__, similarity)
                result[i][j] = similarity
        return result
 
    def _randomize_all_layer_weights(self, model):
        for layer_index in range(len(model.layers)):
            self._randomize_layer_weights(model, layer_index)

    def _randomize_layer_weights(self, model, layer_index): 
        weights = model.get_weights()

        backend_name = K.backend()
        if backend_name == 'tensorflow': 
            k_eval = lambda placeholder: placeholder.eval(session=K.get_session())
        elif backend_name == 'theano': 
            k_eval = lambda placeholder: placeholder.eval()
        else: 
            raise ValueError("Unsupported backend")

        layer_weights = weights[layer_index]
        weights[layer_index] = k_eval(glorot_uniform()(layer_weights.shape))

        model.set_weights(weights)

class KipoiSanityChecker(BaseSanityChecker):
    def __init__(self, kipoi_model):
        self.model = kipoi_model.model

    def _randomize(self):
        pass

    def evaluate(self, similarity_metric):
        pass        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SanityChecker')
    parser.add_argument('--model_path', type=str, help='Path to model')
    parser.add_argument('--sanity_check_type', type=str, default='random',help='type of the sanity check to run')
    parser.add_argument('--explanation_methods', type=str, default='gradient', help='list of explanation methods')
    parser.add_argument('--similarity_metrics', type=str, default='correlation', help='list of similarity metrics')
    parser.add_argument('--batch_size', type=int, default=10, help='')
    args = parser.parse_args()
