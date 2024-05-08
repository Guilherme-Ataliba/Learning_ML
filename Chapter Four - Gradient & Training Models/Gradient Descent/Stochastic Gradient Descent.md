Stochastic gradient descent comes to solve a problem of batch gradient descent, that it uses the whole training set at every step, which makes it very slow on large datasets. SGD solves this problem by just picking **a single random instance** in the training set each iteration.

On the other hand, due to its stochastic (i.e. random) nature, this algorithm is much less regular than batch gradient descent, the cost function will bounce up and down, decreasing on average. 
- Over time, it will end up very close to the minimum, but once it gets there, it'll continuously bounce around it. 
- Once the algorithm stops, the final parameters are good, but not optimal.

# 🔵 The Algorithm
## 🔷 Randomness
⭐ When the cost function is very irregular, this can help the algorithm jump out of local minima. So stochastic gradient descent **has a better chance of finding the global minimum** than batch gradient descent. 
- Randomness is good to escape from local minima, but bad because it means the algorithm never settles. 

## 🔷 Learning Schedule
A solution to this dilemma is to gradually reduce the algorithm learning rate. 
- The steps start out large, which helps make quick progress and escape local minima
- Then it gets smaller and smaller, allowing the algorithm to settle at a global minimum. 

The function that determines the learning rate at each iteration is called ***learning schedule***. There are two points to consider:
1. If the learning rate is reduced too quickly you may get stuck in a local minimum, or even end up frozen in place
2. If the learning rate is reduced too slowly, you may jump across the global minimum and end up with a suboptimal solution. 


## 🔶 Algorithm
```python
n_epochs = 50
t0, t1 = 5, 50

def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2, 1)  # Random initialization

for epoch in range(n_epochs): # This actually counts as the iteration
    # This is equivalent to the matrix operation on the whole dataset
    for i in range(m):
        random_index = np.random.randint(m)
        # use from:to to get an array instead of element.
        xi = X_b[random_index:random_index+1]  
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch*m+i)
        theta = theta - eta * gradients
```

Here we use two loops, the inner of is responsible for running the algorithm (once) on random points of the dataset (100 times). The outer one is responsible for running the algorithm multiple times to increase accuracy, even so that the learning schedule gets smaller, pinpointing on a location as the epochs increase. 

⭐ This algorithm works because it tries to optimize for an instance at a time, and at the end, on average, the algorithm converges to the optimal solution for the dataset. 