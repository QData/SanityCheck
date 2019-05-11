# Sanity Checker: Evaluations for Deep Neural Network Interpretability Tools 

**What:** A Python library as a novel solution to allow researchers to easily conduct evaluations on state-of-the-art interpretability tools. This library currently provides support for Keras and includes an example MNIST model run in the _example_ folder. Our library follows a simple structure: we combine an explanation, a sanity check method, and a similarity metric to make an informed evaluation. 

**Why:** Significant work has been done on interpretability in machine learning, but there is little agreement on how to measure the effectiveness of interpretability tools and evaluate them for benchmarking.

**How:** The central idea in evaluating how “meaningful” a given explanation is revolves around observing how the explanation changes with respect to a change in the model or a change in the data the model was trained on. If we make many changes to the weights of the model and the resulting explanation does not change as much, we can be sure that the explanation is not fit for tasks depending on model weights such as debugging the output class.

**Presentation Video:** https://youtu.be/5rZddtGaq-k

**Presentation Slides:** https://docs.google.com/presentation/d/116IU7noWCZDsb6I8BxPJ4GvFPRGWlrJOiZ1gaMG7sO0/edit?usp=sharing
