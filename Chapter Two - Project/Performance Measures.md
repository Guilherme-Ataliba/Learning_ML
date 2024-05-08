This is how you'll measure how well your model is performing. This metric can represent a lot o things, like the error, accuracy etc. There are some typical measures for regression and classification problems

# 🟠 Convention
* **m**: The number of instances in the dataset you're measuring.
* $\boldsymbol{\vec{x}^{(i)}}$: A vector of all feature values (excluding the label) of the i$^{th}$ instance in dataset
* $\boldsymbol{y}^{(i)}$ Is a vector of all label values (desired output) of the i$^{th}$ instance.
* **X**: is a matrix containing all the feature values (excluding labels) of all instances in the dataset.
	* There is one row per instance, and i$^{th}$ row is equal to the transpose of $\vec{x}^{(i)}$.
* $h$: Is the model's prediction function. Thus, the predicted value can be calculated as $\hat{y}^{(i)} = h(\vec{x}^{(i)})$
For example:

| ![[Pasted image 20230726155341.png]] | ![[Pasted image 20230726155354.png]] |
| - | - |


# 🔵 Regression Problems


## 🔷 Root Mean Squared Error (RMSE)
As the name suggest, it's the median of the sum of error for every instance in the dataset - thus you need to calculate every instance using the predicted function.

* Higher weight on large errors (if too many outlier = bad)
* Error units = label units
* **Norm** - Difference of vectors

$$RMSE(X, h) = \sqrt{\frac{1}{m_i}\sum_{i=1}^m\left(h\left(\vec{x}^{(i)}\right) - \vec{y}^{(i)}\right)^2}$$


## 🔷 Mean Absolute Error (MAE)
* Lighter weight on large errors (best if too many outliers)
* Error units = (label units)$^2$ 
* **Norm** - Difference of vectors 
 
