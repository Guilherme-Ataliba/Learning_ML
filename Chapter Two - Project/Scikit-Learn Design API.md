	## Estimator 
Any object that can estimate some values given a dataset. The estimation is performed by the method fit 
* `fit()` takes as parameter a single dataset in most cases, or two datasets in supervised learning algorithms - where the second one refers to the labels. 

Other parameter used in the estimation processes are the **hyperparameters**. 


## Transformers
They're estimators that also can perform transformations in the dataset. The transformation is performed by the `tranform()` method. 

The transformation usually relies on learned parameters. Transformers also have a convenience method called `fit_transform()` that perform both operations at once and - usually - are optimized to run faster. 


## Predictors 
These are estimators capable of making predictions given a dataset. For example, the `LinearRegression` model predictor. 

A predictor as a `predict()` method that takes a dataset of new instances and returns a dataset of corresponding predictions. 
* It also has `score()` method that measures the quality of the predictions given a test set. 


## Inspection
Hyperparameters are accessible via public instance variables (e.g., `imputer.strategy`) 

all estimator's learned parameters are also accessible via public instance variables with underscore suffix (e.g., `imputer_statistics_`)





