To implement batch gradient descent, you need to compute the gradient of the cost function in regard to each model parameter $\theta_i$. As with most cases in machine learning, it is best if you can also express it in matrix form. 

# 🟠 MSE

### 🔸 1. Calculate the gradient of the cost function
As an example, let's calculate the gradient of the MSE cost function.
$$MSE(\theta) = \frac{1}{m}\sum_{i=1}^m \left(\pmb{\theta}^Tx^{(i)} - y^{(i)}\right)^2$$
The partial derivative with respect to $\theta_i$ is then
$$\frac{\partial}{\partial \theta_j}MSE(\theta) = \frac{2}{m}\sum_{i=1}^m \left(\pmb{\theta}^Tx^{(i)} - y^{(i)}\right)x_j^{(i)}$$
- To figure this out just remember that the derivative will only be different from zero on the $j$ component of $\pmb{\theta}^T$.

Or, in matrix notation
$$\nabla_{\theta}MSE(\theta) = \begin{pmatrix}\frac{\partial}{\partial \theta_0}MSE(\theta) \\ \frac{\partial}{\partial \theta_1}MSE(\theta) \\ ...\\ \frac{\partial}{\partial \theta_n}MSE(\theta)\end{pmatrix} = \frac{2}{m}\pmb{X}^T(\pmb{X}\pmb{\theta} - \pmb{y})$$
- Since $\theta$ ranges over all entries in the dataset, when you calculate the gradient (for every dimension/entry in the dataset) the $x_j^{(i)}$ can just as well be expressed as $\pmb{X}^T$ in the dot product. 

### 🔸 2. Walk in the opposite direction
Once you have the gradient vector, which points in the direction of steepest increase, go in the opposite direction to go downhill. 
- This means subtracting a step of length $\eta$ in the $-\nabla_\theta$ direction from $\pmb{\theta}$.

$$\pmb{\theta}^{(\text{next step})} = \pmb{\theta} - \eta\nabla_{\theta}MSE(\theta)$$
Now you just need to repeat it iteratively until the error gets as small as you need. 

## 🔶 Implementation



### 🟢 Batch
This version of gradient descent is called ***batch*** gradient descent because the formula involves calculations over the full training set $\pmb{X}$, at **each** Gradient Descent step! That is, it uses the whole ***batch*** of training data at each step. 
- As a result it is terribly slow at very large training sets
- However, gradient descent scales well with the number of features.

