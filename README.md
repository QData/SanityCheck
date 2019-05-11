# qdata4kipoi

Sanity Checks for Saliency Maps

Significant work has been done on interpretability in machine learning, but there is little agreement on how to measure the effectiveness of interpretability tools and evaluate them for benchmarking. We present Sanity Checker, a Python library as a novel solution to allow researchers to easily conduct evaluations on state-of-the-art interpretability tools. 

This library currently provides support for Keras. An example MNIST model run is provided in the examples folder.

The central idea in evaluating how “meaningful” a given explanation is revolves around observing how the explanation changes with respect to a change in the model or a change in the data the model was trained on. Our library follows a simple structure: we combine an explanation, a sanity check method, and a similarity metric to make an informed evaluation.
