Training Module
==========================================

The training module contains an entrypoint for model training and saving 
trained model artifacts. This is essentially a script inside of a function 
called `handler` that steps through each process in training.  

It's important to split critical functionality into classes and 
functions, so that unit tests can be created to test individual bits 
of functionality that you've created; while, tests for `handler` only 
check that each known step is executed (e.g. load data, create model, fit, dump).


.. autosummary::
    :toctree: modules
    :recursive:

    production_demo.train
    production_demo.train.handler
    production_demo.train.CategoriesTransformer
    production_demo.train.CategoriesTransformer.hash_col
    production_demo.train.CategoriesTransformer.transform
