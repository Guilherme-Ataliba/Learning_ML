# 🔵 Linear Models
Make predictions by computing a **weighted sum** of the input variables, plus a constant term called **bias**:
$$\hat{y} = \theta_0 + \theta_1x_1 + \theta_2x_2 + ... + \theta_nx_n$$
- Where the $\theta_i$ is the i$^{th}$ **model parameter**.

This can be written in a vectorized form:
$$\hat{y} = h_{\theta}(x) = \pmb{\theta} \cdot \pmb{x}$$
- $\pmb{\theta}$: Is the model's parameter vector
- $\pmb{x}$: Is the instance's features vector
- $h_{\theta}$: Is the hypothesis function, using the model parameters $\pmb{\theta}$ 

### 🔹 Performance Measure
In this case, the most common one used is the Root Mean Squared Error. Thus, we have the problem to find the $\pmb{\theta}$ that minimizes RMSE.

❗ In practice is best to minimize the **MSE**, since it has lest steps in the calculation it is faster to be evaluated and, most importantly, it is a **quadratic function**, thus, it will have minimum value.


The MSE of a linear regression hypothesis $h_{\theta}$ on a training set $\pmb{X}$ is calculated using
$$MSE(\pmb{X}, h_{\pmb{\theta}}) = \frac{1}{n}\sum_{i=1}^n\left(\pmb{\theta}^T{x}^{(i)} - y^{(i)}\right)^2$$

#### - Column Vectors
Machine learning vectors are often represented as **column vectors**, which are **2D arrays** with a single column. If $\pmb{\theta}$ and $\pmb{x}$ are column vectors, then the dot product can be expressed as
$$\hat{y} = \pmb{\theta}^T \pmb{x}$$
- This notation infers that the prediction is a one cell matrix instead of a scalar. 


--- 
## 🔷 The Normal Equation
This is a closed-form mathematical equation that gives the value of $\pmb{\theta}$ that minimizes the cost function:
$$\hat{\theta} = \left(\pmb{X}^T\pmb{X}\right)^{-1} \pmb{X}^T \pmb{y}$$
- $\hat{\pmb{\theta}}$: Is the predicted value of $\pmb{\theta}$ that minimizes the **cost function**.
- $\pmb{y}$: Is the vector of target values.



## 🔷 Implementation

#### Training
After you've got the data (X and y), the algorithm is simple:
```python
X_b = np.c_[np.ones((100, 1)), X]
theta_best = np.linalg.inv( X_b.T.dot(X_b) ).dot( X_b.T.dot(y) )
```
1. We add a column of ones to the dataset, this is so the algorithm can calculate the bias term.
2. Use the normal equation to calculate the thetas.

#### Predictions
To make predictions, you just need to calculate the dot product of $\hat{\pmb{\theta}}$, calculated above, and the input data (don't forget to put in the right format and add the column of ones).
```python
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_pred = X_new_b.dot(theta_best)
```


## ❗ Inverse Matrix
The inverse matrix is not always defined and can be a hassle to compute. To fix both of these problems, instead of calculating the inverse matrix, we could calculate the pseudo-inverse (Moore-Penrose inverse) matrix $\pmb{X}^{+}$.
- The pseudo-inverse matrix always exists 
- Utilizing Singular Value Decomposition (SVD) the computational time is reduced to $O(n^2)$, in comparison to the original inverse that is $O(n^3)$. 

In NumPy we can directly use the method `np.linalg.pinv(X_b)`.

