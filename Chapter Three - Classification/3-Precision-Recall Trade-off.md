The precision recall trade off states that: *Increasing precision reduces recall, and vice versa.*

# 🔵 Threshold & Decision Function
Classifier algorithms work with a **decision function**, that, for a given set of values, returns a number. Now, that number is compared against a threshold - meaning that - with it is greater than that threshold it's considered **positive** and if it is lower it's set to **negative**.

![[Pasted image 20230912214727.png]]
This image show how the precision-recall trade-off works:
1. If you start with the threshold at the middle, there are 4/5 right predictions (80% precision), but out off 6 5's the algorithm only detected 4 (67% recall)
	
2. If you increase the threshold the number of false positives reduces to zero, 3/3 fives (100% precision), but one 5 was left out, so out of 6 5's only 3 were detected (50% recall)
	
3. If you decrease the threshold, the algorithm detects 6 5's out of 6 (100% recall), but out of 8 detection only 6 were right (75% precision)


💡 **Recall** can only go down when the threshold increases.
💡 *Usually* **precision** goes up when the threshold increases, but that is not a rule.

## 🔷 Getting the Decision Function
Scikit-learn does not allow you to set the threshold directly, but there's a way of getting the *decision function* used in the **trained** classifier:
```python
y_scores_function = sgd_clf.decision_function
y_scores = y_scores_function([some_digit])

threshold = 0

y_pred = y_scores > threshold
y_pred
```
From the decision function, you may calculate the score of each entry and manually set a threshold and a verification process (like above).


# 🔵 Finding the Best Threshold
We can analyze graphically what the best combination of precision and recall we could get for that specific problem (since there isn't a general best, you must find the one the better fits your problem).

We're able to get 3 quantities as arrays - and then plot from them what we want:
1. **Precisions**: An array that contains the precision for different values of thresholds
2. **Recalls**: An array with the recalls for different values of thresholds.
3. **Thresholds**: The different values of thresholds used.

To get them, use:
```python
from sklearn.metrics import precision_recall_curve

# Get the scores from the decision function
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")

# Get the quantities
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
```

#### "Setting" Precision / Recall
You can search for a specific value of precision/recall and what is the required threshold for you to get that value. To do that, use NumPy's `argmax` method:
```python
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
```

## 🔵 Different Plots

#### Precision & Recall X Threshold
![](graph-1.png)

#### Precision X Recall
![](graph-2.png)
* This kind of graph is specially useful to compare clearly how one affects the other
	* For example, precision reduces greatly after 0.8 recall, so that could be a value that we don't go past. 

The closer this curve is to the top-right corner the better the classifier.

## 🔷 Getting Specific Precision Algorithms
To do this, you'll just have to set the threshold manually, for example:
```python
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]

y_train_pred_90 = (y_scores >= threshold_90_precision)

precision_score(y_train_pred_90)
```


# 🔵 ROC Curve

## 🔷 More Metrics
Receiver Operating Curve, it plots **recall** against **1-specificity (FPR).
* **Recall:** Also called true positive rate or **sensitivity**.

#### False Positive Rate
Is the ratio of negative instances that are incorrectly classified as positive. 
$$FPR = \frac{FP}{FP + TN}$$
> It's the probability of a false alarm to be raised

#### True Negative Rate - Specificity
The ratio of positive instances that are correctly classified as negative. 
$$TNR = \frac{TN}{TN+FP}$$
> Suppose a medical test has specificity of 95%, this means that if 100 people who do not have the disease take the test, the test will correctly identify 95 of them

## 🔷 The ROC Curve
The ROC curve shows the **True Positive Rate** against the **False Positive Rate**, it means that, the more your graph is close to the top-left corner, the better is your model. 

Since you want to maximize the **TPR** (y-axis) and minimize the **FPR** (x-axis).
![[Pasted image 20231011165744.png]]
* The closer the ROC curve is from the top-left corner, or, the farthest away it is from the dotted linear curve the better. 
* We want a really high value that has almost none x value.
* You would want to get values of threshold that gives you high recall and low FPR, thus, values before the curve stays almost horizontal

The algorithm starts very good, with small values of FPR. But increasing FPR increases recall very slowly - which is not useful.

## 🔷 AUC - Area Under ROC Curve
This is a measure of how good a model is, a perfect classifier would have $AUC=1$, and a purely random classifier (dotted line) would have an $AUC=0.5$.



## 🔷 Dict Probability Algorithms
Some methods in scikit-learn doesn't have scores - used to get the ROC curve, some of them have a `dict_proba()` method, that returns the probability that a given instance belongs to a given class. And from that probability you classify the instance.

One example of this is the `RandomForestClassifier`. You can get the prediction probabilities by:
```python
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")

y_probas_forest
```

But for the ROC curve you need scores, not probabilities. One solution is to use the probability of the positive class as the score:
```python
y_scores_forest = y_probas_forest[:, 1]
```

Then you can get the ROC curve just as usual. 

### 🔷 Compare the ROC curve with the Precision Recall Curve
You must understand what you want for your classifier and what is more important. Let's see compare the two for two different classifiers:

![[Pasted image 20231011174259.png]]
We can clearly see the **Random Forest Classifier** performed better for the ROC curve (better on top-left corner) and the precision recall curve (better on top-right corner). 
