Are a method for visualizing how well a model performs (cost function) with respect to how many items there are in the training set. This kind of graph can show a lot of interesting information, especially with respect to characterization of **overfitting** and **underfitting**. 

Learning curves plot the cost function value for different training set sizes. The algorithm looks something like:
```python
def plot_learning_curves(model, X: np.array, y: np.array):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors : list[float] = []
    val_errors: list[float] = []
    
    # Rodar adicionando mais items no dadaset a cada iteração
    for m in range(1, X_train.shape[0]):
        X_filt, y_filt = X_train[:m], y_train[:m]
        model.fit(X_filt, y_filt)
        
        # Há duas curvas, uma para o treino outra para o teste
        y_train_pred = model.predict(X_filt)
        y_train_val = model.predict(X_val)
        train_errors.append(mean_squared_error(y_filt, y_train_pred))
        val_errors.append(mean_squared_error(y_val, y_train_val))
    
    plt.plot(np.sqrt(train_errors), "r-")
    plt.plot(np.sqrt(val_errors), "b-")
```
The idea is to simply train the model multiple times for different sizes of the training set and see how the error function changes.
- This process can be done iteratively, as shown above. 

# 🔵 Reading the Curves
Let's see some examples of learning curves and what they mean./

## 1. 
![[Pasted image 20240228182342.png]]

#### Small training set size
- For a few points in the dataset is very easy to train the model to fit perfectly the data, that's why for very low entries the training error tends to zero. 
- But, with just a little training points the algorithm generalizes very poorly, meaning it is very prone to overfitting. And that's why the validation error starts very large

#### Medium and high training set size
- After the start, both curves tend to reach a plateau, meaning that adding more training points makes almost no difference in the algorithm's performance. 

#### Conclusion
This is clearly a case of **underfitting**, since both curves have fairly high RMSE and adding more training instances makes almost not difference. 
- This is common for curves that reach this kind of plateau.

This curve show that if a model is underfitting the data, adding more points will not help algorithm's performance. One needs to use a more complex model or come up with better features. 


## 2. 
![[Pasted image 20240228183039.png]]

#### Small training set size
The discussion here is very similar to the one before with respect to small training instances.

#### Medium and high training set size
- The final error of the algorithm is lower than the previous one. This means in general that this model fits the data better.
- There's a gap between the curves, that means:
	1. The model performs significantly better on the training set than the validation set, which is a hallmark that the model is **overfitting** the data.
	2. The more training instances are used more the validation curves approximates the training curve.

#### Conclusions
The fact the adding more training instances increases validation accuracy shows a characteristic of **overfitted models**: If you increase the number of training instances, the overall performance will improve.

One way to improve an overfitting model is to feed it more training data until the validation error reaches the training error.
- This means that there is so much training data that, even if the algorithms just memorize it, it is still so large that the validation problems are included in the training set, in some way or another.


# 🔵 Bias/Variance Trade-off
A model's generalization error can be expressed as a sum of three very different errors:

#### Bias
This part of the error is due to wrong assumptions (such as a linear model for a quadratic data). High-bias models are likely to **underfit** the training data. 

#### Variance
This part is due to the model's excessive sensitivity to small variations in the training data. Usually related to very complex models with a high number of degrees of freedom. Models with high variance are prone to **overfitting** the data. 

#### Irreducible error
This part is due to the noisiness of the data itself. The only way to reduce this part of the error is to clean up the data.


#### 🟢 The Trade-off
- Increasing a model's complexity will increase its variance and decrease its bias.
- Decreasing a model's complexity will increase its bias and decrease its variance.

This is why it is called a trade-off. 