# 🔵 Final/Model Pipeline

An ML project consists of a lot of different steps, specially referred to data preparations:

1. Treat missing values
2. Feature scaling
3. Selecting features
4. Adding new features (combinations) 
5. Treat categorical features
6. Apply to a model

Each of these steps has hyperparameters to be tuned (from tuning the model's hyperparameter to choosing how feature scale and which features to add). 

The process of hyperparameter optimization can be made with the help of a **grid search**, but, to be able to tune all of them at once it is required that all processes are in a single pipeline
* Besides, creating a single pipeline with all process required - from data preparation to fitting the model - makes the code much cleaner.


## 🔷 Creating the Pipeline
Be sure that all functions that all you need to do that refers to data preparation is inside a method or a function you've created. If all is done, you just need to create the pipeline with all method, in order of execution.

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


housing_ready = housing.copy()
housing_ready.drop("median_house_value", axis=1, inplace=True)

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),   # fixing missing values
    ("attribs_adder", CombinedAttributesAdder()),    # Adding the combined features
    ("std_scaler", StandardScaler())                 # Fixing the features' scales
])


# Defining categorical and numerical attributes
cat_attribs = list(housing["ocean_proximity"].unique())    
num_attribs = [x for x in housing_ready.columns if x not in cat_attribs]
num_attribs.remove("ocean_proximity")

# Encoding categorical features outside the pipeline
one_hot_encoder = OneHotEncoder()
transformed = one_hot_encoder.fit_transform(housing_ready[["ocean_proximity"]])

housing_ready[one_hot_encoder.categories_[0]] = transformed.toarray()
housing_ready.drop("ocean_proximity", axis=1, inplace=True)
display(housing_ready)

# Transformation Pipeline
transformation_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", "passthrough", cat_attribs)
])

# Setting the random_state for better testing
forest_reg = RandomForestRegressor(random_state=42)

# Creating the model pipeline
forest_pipeline = Pipeline([
    ("transformation", transformation_pipeline),
    ("model", forest_reg)
])
```
Two classes were used:
1. `Pipeline`: This is the base class to create a pipeline. 
2. `ColumnTransformer`: This works just like a pipeline, except it allows for applying transformations on a specific column. 

Observations:
* Pipelines can and were combined in the above example. 
* The names given (inside quotation marks) are very important for hyperparameter tuning. 


## 🔷 Hyperparameter Tuning
To implement this, we'll use `GridSearchCV`. The only difference from tuning the hyperparameters of a usual model is in the declaration of the names of each parameter:

To "select" the hyperparameter name to pass to a grid search, that is inside a pipeline, you must write the *"path"* to that hyperparameter. 
* **Path**: Consists of the names given in the pipeline creation, in order, from outside in, until you get to the hyperparameter.

For example, using the previous `forest_pipeline`, the parameter grid from the `GridSearchCV` becomes:
```python
param_grid = [
    {"model__n_estimators": [3, 10, 30], 'model__max_features': [2, 4, 6, 8], "transformation__num__attribs_adder__add_bedrooms_per_room": [False, True]},
    {'model__bootstrap': [False], 'model__n_estimators': [10, 30], 'model__max_features': [4, 6, 8], "transformation__num__attribs_adder__add_bedrooms_per_room": [False, True]}
]

grid_search = GridSearchCV(forest_pipeline, param_grid, cv=5, scoring="neg_mean_squared_error", return_train_score=True)

grid_search.fit(housing_ready, housing_labels)
```

### 🔺 Observation 1 - Creating Features in a Pipeline
When you create/add a feature to the data inside a pipeline, the attributes 
* `grid_search.n_features_in_`
* `grid_search.feature_names_in_`
Will not count the added feature (just the ones given as input to the grid search) - even though it'll affect the results and **will be** implemented. 

To see the actual number of features you can use:
```python
# change "transformation_step" to your used name
transformation_step = grid_search.best_estimator_.named_steps["transformation_step"]

transformation_step.transform(X).shape[1]
```


### 🔺 Observation 2 - Encoding Categorical Features
We need to apply the one hot encoder first because it'll split the categorical data accordingly to the available data in the subset. Sometimes it happens that the grid search CV takes a subset where one of the categories is not present, and that makes a problem, when it compares to the general expected features.

So, to fix that problem, the encoding of categorical features must be done separately from the pipeline.