Scikit-Learn takes as input in its methods `X` (big x) and `y` (small y) data. `X` refers to the training data and `y` the labels. 
* `X` Is a 2D array or sparse matrix 

# 🔵 Transforming Pandas Data Frame → X
There are two major ways:
1. `dataframe_name.values`
2. `np.c_[dataframe_name]`

It is also possible to use `np.c_[]` to add new columns to the 2D matrix, like so:
```python
rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        
np.c_[X, rooms_per_household, population_per_household]
```