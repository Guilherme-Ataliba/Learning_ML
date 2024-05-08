It is a very generic optimization algorithm capable of finding optimal solutions to a wide range of problems. The general idea of Gradient Descent is to tweak parameters iteratively in order to minimize a cost function. 

# 🔵 How it works
Gradient descent measures the local gradient of the error function, with regard to the parameter vector $\theta$, and it goes in the direction of descending gradient. Once the gradient is zero, you've reached a minimum. 
- It works by calculating how "steep is the slope". It calculates the direction something decreases the most, and it goes in that direction. 

![[Pasted image 20240219204004.png|500]]

#### Operationally 
0. **Random Initialization**: Start by filling $\theta$ with random values.
	- This is an important step, since the starting point may generate very different answers. 
1. Calculate the gradient at that point.
2. Take a step of length $\eta$ in the gradient's opposite direction
	- Small steps make the algorithm more precise (sometimes), and usually makes it take more time to finish.
3. Repeat steps 1-3 until the gradient approaches zero. 

The size of steps is determined by the ***Learning Rate*** hyperparameter. If the learning rate is too small, then the algorithm will have to go through many iterations to converge, which will take a long time.
![[Pasted image 20240219204407.png|500]]
- It may also cause it to be stuck on a local minimum.

On the other hand, if the learning rate is too high, you might jump across the valley and end up on the other side. This might make the algorithm diverge, failing to find a good solution. 
![[Pasted image 20240219204547.png|500]]

## 🔷 Difficulties
Gradient descent is very dependent on the cost function, some of them are really hard to calculate the global minimum (and most of the time you won't know if the answer you've got is in fact the local minimum). The next figure shows the two most common problems
![[Pasted image 20240219204921.png|500]]
1. If the algorithm starts on the left, it may converge to a local minimum, instead of the global minimum. 
2. If the algorithm starts on the right, then it'll take a very long time to come across the plateau, and if you stop too early you'll never reach the global minimum. 

## 🔷 Cost Functions
Two conditions on the cost function guarantees that the gradient descent will converge, if the learning rate is not too high:
1. **Convex functions**: They have no local minima and just one global minimum. 
2. **Continuous (well-behaved) functions**


## 🔷⭐ Scale
When using Gradient Descent, you should ensure that all features have a similar scale, otherwise it will take much longer to converge. 
- This is why we scale data for RNA algorithms. 

