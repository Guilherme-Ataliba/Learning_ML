# 🔵 Any Regression
We may use a linear model to fit any kind of function to a dataset. This includes, but is not limited to:
1. Polynomials (polynomial regression)
2. Exponential
3. Logs
4. Square root
5. Trigonometric functions
6. Any combination of the above

The only problem is we have to define the function beforehand, that means, define how the function is, and the algorithm will find the best parameters. 

## 🔷 Process
We may use any linear model technique (normal equation, gradient descent, etc.). After that, one must define the form of the function, for example:
$$f(x) = ax^2 + be^x - cx\sin(x) + d$$
Then, you'll need to create a new column in the X dataset for every new term in the function. Each column will correspond to one of the terms applied to the original data. 
- For example, the following code adds a new column that expresses the first term of the $f(x)$ function.
- This procedure must be done for every term on the function (this includes a column of ones). 
```python
np.c_[X, X**2]

# This must be done for every term on the function
X_new = np.c_[X, X**2, np.exp(X), np.sin(X)*X] 
```

After that, you simply fit it to a linear model, and it'll find the appropriate value for the constants. 

## 🔷 Explanation
This works because when you transform the columns of the entry data, it is like you've changed the "space" for the linear model. This means that from his perspective it is still all a simple linear equation, that it knows how to fit. 

#### Code
![[LinearRegression.ipynb]]