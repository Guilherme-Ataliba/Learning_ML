Here we'll have a look a different ways and metrics to evaluate your classifier.


# 🔵 Decision Boundary
In classification algorithms the model will have what is called a **decision boundary**, this boundary defines what the algorithm will classify as different values.

For example, in a binary classifier, we got this decision boundary
![[Pasted image 20230901165405.png|600]]
* Bellow the line (white background) everything will be classified as 0
* Above the line (blue) everything will be classified as 1

For this specific case, the model used is not good for the problem in question, since the data as clearly a exponential shape and the model express a linear curve - *decision boundary*.

# 🔵 Implementing Cross Validation 
Occasionally we'll need more control over the cross-validation processes than what is offered by scikit-learn, to fix this we can implement our own version of cross-validation:

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

def my_cross_validation(model, X, y, n_splits=3, shuffle=False, random_state=False):
    
    if (random_state != False):
        skfolds = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    else:
        skfolds = StratifiedKFold(n_splits=n_splits, shuffle=shuffle)

    results = []
    for train_index, test_index in skfolds.split(X, y):
        model_clone = clone(model)
        
        X_train_fold = X.loc[train_index]
        X_test_fold = X.loc[test_index]
        y_train_fold = y.loc[train_index]
        y_test_fold = y.loc[test_index]
        
        model_clone.fit(X_train_fold, y_train_fold)
        
        y_pred = model_clone.predict(X_test_fold)
        
        n_correct = sum(y_pred == y_test_fold)
        results.append(n_correct/ len(y_pred))
    
    return results
```
* The `skfolds.split()` method returns the indexes to use to choose the rows of the dataframe.
* It is important to make a stratified sampling to guarantee that every class is represented. 


# 🔵 Accuracy
Is calculated as the ratio of correct predictions divided by the number of predictions. 
$$\text{accuracy} = \frac{\text{right predictions}}{\text{number of predictions}}$$
```python
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
```
The result is apparently good, but in fact, what is happening is that accuracy is a bad metric for this case. 
* Less than 10% of the data consists of 5s, thus, what the classifier actually does is not relevant, since 90% accuracy is already guaranteed. 


## 🔷 Baseline
The baseline is a simple model that you use as a metric to measure how bad your algorithm is. If your model is close to the baseline (or worst), it's not a good model and you must improve **a lot**.

Usually in **binary classification problems,** the baseline will be a "guess all ones/zeros" kind of model. To implement this, you only need to fill an array with ones/zero and calculate its accuracy.
```python
from sklearn.metrics import accuracy_score

baseline = np.ones(predictions.shape[0])
accuracy = accuracy_score(y_test, baseline)
accuracy
```


#### 🔴 Conclusions
* Accuracy is a bad metric for skewed datasets - where some classes are much more frequent than others.


## 🔷 Dummy Classifier / Regressor
Dummy classifier/regressors are Scikit-learn's built in models to make **baseline predictions** in an automate manner – and with a lot of options. 

You must first import and instantiate a dummy classifier/regressor
```python
from sklearn.dummy import DummyClassfier

dummy = DummyClassifier(strategy="__strategy", random_state=)
```

#### Strategy
Refers to how the dummy classifier will “predict” the new values, some options are:
* `most_frequent` Repeats the most frequent value (only true/false)
* `stratified` Preserves the proportion of the input data and guesses that proportion as prediction

For more information, click [here](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html).

#### Predictions / Accuracy
After that, you can just predict or get the accuracy directly from the model
```python
dummy.fit(X_train, y_train)
predictions = dummy.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# OR

dummy.fit(X_train, y_train)
accuracy = dummy.score(X_test, y_test)

```

--- 
# 🔵 Confusion Matrix
Each row in a confusion matrix represents an *actual class* (the actual values), while each column presents a *predicted class* (the predicted values
![[Pasted image 20231014184323.png]]
![[Pasted image 20230825171822.png]]
* FF: Values that were **false** and were predicted as **false**
* FT: Values that were **false** and were predicted as **true**
* TF: Values that were **true** and were predicted as **false**
* TT: Values that were **true** and were predicted as **true**

Thus, in a confusion matrix, we want to maximize the top-left corner and the bottom-right corner. 
* A perfect classifier would have zero everywhere else. 

The elements of a confusion matrix have actual names, they are:
![[Pasted image 20230825172238.png]]
* True Negatives (**TN**)
* False Positives (**FP**)
* False Negatives (**FN**)
* True Positives (**TP**)

## 🔷 Precision and Recall
The confusion matrix gives a lot of information, and to make it clearer to work with it, there are some metrics we can get from it.

### 🔹 Precision
The accuracy of the positive predictions:
$$
\text{precision} = \frac{TP}{TP + FP}
$$
* What were correctly classified as positive divided by the total amount classified as positives.

A way to get perfect precision would be to make one single positive prediction and make sure it is right, thus 1/1 correct predictions. This means that precision by itself is not very useful, we must know the rate of positive predictions - this is called **sensitivity**. 


> 🔸 **Example**: A precision of 72.5% means that when your classifier claims a value is 1 (true) it is right 72.5% of the time


### 🔹 Recall / Sensitivity 
The ratio of positive instances that are correctly detected by the classifier. 
$$
\text{recall} = \frac{\text{TP}}{\text{TP + FN}}
$$
* This calculates the ratio of what the classifier judge as true and the total number of actually true values. 

> 🔸 **Example**: A recall of 75.6% means that your classifier only detects 75.6% of the 1 (true) instances as true.


### 🔸 Scikit-Learn Implementation
Scikit-learn provides functions to calculate precision and recall:
```python
from sklearn.metrics import precision_score, recall_score

display(precision_score(y_train_5, y_train_pred))
recall_score(y_train_5, y_train_pred)
```


### 🔹 $F_1$ Score
Is a metric that combines precision and recall - it can be used to compare two classifiers. $F_1$ score is the *harmonic mean* of precision and recall.

#### Harmonic Mean
Whereas a regular mean treats all values equally, a harmonic mean gives much more weight to low values. This means that the harmonic mean will only be high if both precision and recall are high. 

For a number **n** of observations $\boldsymbol{x_i}$, the harmonic mean is given by
$$
\text{harmonic mean} = \frac{n}{\displaystyle \sum_{i=1}^n \frac{1}{x_i}}
$$
Thus, the $F_1$ score can be calculated by
$$
F_1 = \frac{2}{\frac{1}{\text{precision}} + \frac{1}{\text{recall}}} = 2 \cdot \frac{\text{precision}\cdot\text{recall}}{\text{precision}+\text{recall}}
$$

### 🔹🔸 Discussion About Metrics
The $F_1$ score favors classifiers that have similar precision and recall, and that isn't always what you want. Examples:

1. If you train a classifier to detect videos that are safe for children, you'd favor a classifier that rejects many good videos (**low recall**) but keeps only safe one (**high precision**)
2. Now, suppose you want to train a classifier to detect shoplifters on surveillance images. You'd sure prefer a model that detects many wrong people (**low precision**) but detects all shoplifters (**high recall**).









