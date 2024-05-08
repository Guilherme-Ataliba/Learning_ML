At each step, instead of computing the gradients based on the whole training set (as in batch gradient descent) or based on just one instance (as in stochastic gradient descent), **mini-batch gradient descent** computes the gradients on small random sets of instances called *mini-batches*.

#### Advantage 
The main advantage of **mini-batch gradient descent** over SGD is that you can get a performance boost from hardware optimization of matrix operations, especially using GPUs. 


#### Differences
The algorithm's progress in parameter space is less erratic than SGD, especially with fairly large mini-batches. 
- As a result, mini-batch GD will end up walking a bit closer to the minimum than SGD
- On the other hand, it may be harder for it to escape from local minima, just like BGD.
![[Pasted image 20240221135613.png]]
- Remember that it is possible to make SGD and MGD converge closer to the minimum with a good *learning schedule*. 

#### Conclusion
![[Pasted image 20240221135742.png]]

## 🔶 Algorithm
```python
n_iterations = 50
minibatch_size = 20

theta = np.random.randn(2, 1)

t0, t1 = 200, 1000

def learning_schedule(t):
    return t0 / (t + t1)

t = 0
for epoch in range(n_iterations):
    # shuffle the indices intead of picking one randomly 
    # since it would be a hassle for this amount of elements (its easier)
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    
    for i in range(0, m, minibatch_size):
        t+=1
        xi = X_b_shuffled[i:i+minibatch_size]  # batch size array
        yi = y_shuffled[i:i+minibatch_size]
        gradients = 2/minibatch_size * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(t)
        theta = theta - eta * gradients
```