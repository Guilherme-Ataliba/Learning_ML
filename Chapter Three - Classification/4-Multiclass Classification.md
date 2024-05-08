# 🔵 Extending Binary Classifiers
Some models are inherently capable of doing multiclass classification, like Naive Bayes or Random Forest. Others are not, they're restricted to binary classification problems. 

For those cases, we can implement binary classifier in different ways, to make the resultant model capable of doing multiclass classification.

⭐ Scikit-Learn automatically detects when you try to use a binary classification algorithm for multiclass classification problems, and automatically use OvA or OvO. 

## 🔷 One Versus All (OvA)
A possibility is to train a binary *"detector"* classifier for each different class (for example, a 0-detector, a 1-detector and so on). Then, when you want to classify something, you get the decision score for each classifier and select the class whose classifier outputs the highest score.

**To Note:**
* This means that, for N classes, you must train N classifiers on the whole dataset.
	* Some algorithms scale poorly with the size of the training set - so other alternatives could be better. 


## 🔷 One Versus One (OvO)
In this case we o train a binary classifier for every pair of classes (one to distinguish 0 & 1, other to distinguish 0 & 2, another for 1 & 2, and so on).

If there are N classes, you'd need to train $\frac{N(N-1)}{2}$ classifiers. 
* The number of classifiers increase quickly. 

When you want to classify something, you need to run it through all classifiers and see which one wins the most duels. 

**No Note**:
* The main advantage is that each classifier only needs to be trained on the part of data that refers to its classification
	* This could be useful for some algorithms that scale poorly as data size increases. 


## 🔷 Scikit-Learn Classes
We can manually define which method will be used (OvA or OvO) in a multiclass classification problem with the scikit-learn classes `OneVsOneClassifier` and `OneVsRestClassifier`, just like so:
```python
from sklearn.multiclass import OneVsOneClassifier

ovo_clf = OneVsOneClassifier(SGDClassifer(random_state=42))
ovo_clf.fit(X_train, y_train)
ovo_clf.predict([some_digit])
```
* You just need to pass the model instance to the class you've decided on. 



# 🔵 Multiclass Confusion Matrix
The original confusion matrix can be thought as a confusion matrix of two classes (True and False predicted values), now we must expand this concept to a full size confusion matrix for N classes. It'll look something like this:
![[Pasted image 20231014184738.png]]
* In this case, **0** to **9** are the classes for the classification problem.

## 🔷 Classifications
Just like a 2D confusion matrix, the multiclass confusion matrix has the categories of **TP, FP, TN & FN**, but they refer to a specific class at a time. 

For example, for the first class (**Class 0**), we have:
![[Pasted image 20231014191339.png]]
There's no clear concept as **TN** for a multiclass classification problem, thus, metrics that utilize **TN** can only be defined for the entire class

## 🔷 Class Metrics
We can calculate *class-specific* metrics, such as precision, recall, etc. Just like usual:

$$\text{precision}_0 = \frac{TP_0}{TP_0 + FP_0}$$
$$\text{recall}_0 = \frac{TP_0}{TP_0 + FN_0}$$
$$F_{1_0} = \frac{2}{\frac{1}{\text{precision}_0} + \frac{1}{\text{recall}_0}} = 2 \cdot \frac{\text{precision}_0\cdot\text{recall}_0}{\text{precision}_0+\text{recall}_0}$$

## 🔷 General Metrics
And, as one would expect, we're also able to define these metrics for the whole model - the most general being **accuracy**. 

$$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{All Predictions}}$$
* **Correct Predictions:** The sum of the main diagonal on the confusion matrix
* **All Predictions:** The sum of all values on the confusion matrix



## 🔹 General Precision & Recall
Instead of having multiple precision and recall metrics, one for each class, we can *average* these metrics and get a general one - for the whole data. 


### 🔹 Macro-averaging 
Compute precision and recall for each class and average it by the total number of classes:

$$\text{Macro Precision} = \frac{\text{Precision}_1 +\text{Precision}_2 + ... + \text{Precision}_N}{N}$$
$$\text{Macro Recall} = \frac{\text{Recall}_1 +\text{Recall}_2 + ... + \text{Recall}_N}{N}$$
* Where **N** is the total number of classes


### 🔹 Micro-averaging
First you calculate the **TP, FP & FN** across all classes, meaning, the sum of **TP, FP & FN** of all classes. Then you calculate the precision and recall using these values.

$$\text{Total TP} = TP_1 + TP_2 + ... + TP_N$$
$$\text{Total FP} = FP_1 + FP_2 + ... + FP_N$$
$$\text{Total FN} = FN_1 + FN_2 + ... + FN_N$$
Then, calculate the metrics;
$$\text{Micro Precision} = \frac{\text{Total TP}}{\text{Total TP} + \text{Total FP}}$$
$$\text{Micro Recall} = \frac{\text{Total TP}}{\text{Total TP} + \text{Total FN}}$$


## 🔷 Which Metric to Use?


### 🔹 Macro-averaging X Micro-averaging
Macro averaging calculates each class's performance first and then takes the arithmetic mean - thus it gives **equal weight to each class** regardless of the **number of instances**.


Micro averaging gives equal weight to each instance, this means that if a class has lower instances it'll have lower weight on the resulting metric. 

Which one to use will depend on the situation, but macro-averaging will penalize bad metrics (don't care if it has lower instances) and micro-averaging will give less weight to the metrics of classes that have lower instances. 


### 🔹 Use the Confusion Matrix
One of the most used metrics for multiclass classification problems is the **heatmap of the confusion matrix**. If most of the values are on the main diagonal, this indicates a good model, since those are the values that were **classified correctly**. 

```python
sns.heatmap(confusion_matrix)
```


## 🔷 Error Confusion Matrix
We can plot a confusion matrix with only the errors for each class - thus analyzing the types of errors the algorithm does. 

1. First normalize the confusion matrix, dividing each class by the total number of entries. This will make you able to compare rates of errors, instead of absolutes values
	* Absolute values can be misleading, since classes with a lot of instances will appear unfairly bad
2. Set the main diagonal equal to zero, this way you're able to analyze just the errors. 

```python
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)


sns.heatmap(norm_conf_mx)
```

### 🔹 Conclusions Example
Remember that columns represent predicted values and row actual values.
![[Pasted image 20231015100523.png]]

* In this case, the column of class eight is brighter, which tells you that many images get misclassified as 8.
* But the row of class 8 is OK, which tells you that most 8s get properly classified. 
* 3s and 5s get confused in both directions (rows and columns) 

Analyzing this confusion matrix (the types of errors it commits) can give you direction in ways to improve your algorithms. For example:

1. You should focus on reducing the false 8s. This could be achieved in some ways:
	1. You could gather more data on numbers that look like 8 but are not
	2. Or you could engineer a new feature, by counting the number of closed loops in each figure (8s are the only number with 2 closed loops)
	3. Or you could preprocess the image to make patters stand out more

#### Analyze the individual types of errors
![[Pasted image 20231015102435.png]]
* Some miss classifications even a human would get confused. But most of them are not clear why the algorithm is miss classifying. 


# 🔵 Multilabel Classification 
These are problems that you want your classifier to output more than one class per instance. For example, in a face-recognition classifier if more than one person is on the same image. 

In general, a multiclass classification algorithm outputs multiple binary tags. For example, if your algorithm is trained to classify 3 faces, A, B and C; Then in a picture with A and C it should output `[1, 0, 1]`. 

```python
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf.fit(X_train, y_multilabel)

knn_clf.predict([some_digit])

>>> array([[False,  True]])
```
This creates a multilabel classifier that predicts if a number of greater or equal to 7 & if it is odd. 
* In this case, it predicted 5 as not greater than 6 and that it is odd.  

There are several ways to measure performance for multilabel classification, for example using F1 score for each class and then calculating the total average:
```python
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average="macro")
```


# 🔵 Multioutput Classification
Or multiclass multioutput classification is the most general classification problem, it consists of each label can be multiclass (more than two possible values) and it should return more than one class per instance. 

## 🔷 Example
Let's create a model that removes noise from images. It'll take as input a noisy image and output a clean image. It is a multioutput classification problem since it'll output multilabel (one label per pixel) and each label can have multiple values (pixel intensity from 0 to 255).