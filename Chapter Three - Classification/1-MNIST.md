# 🔵 Import Scikit-learn Famous Datasets
Scikit-learn provides a way of downloading the most famous datasets directly, it is as simple as:

```python
from sklearn.datasets import fetch_openml
mnist = fetch_openml("mnist_784", version=1)
mnist.keys()

>>> dict_keys(['data', 'target', 'frame', 'categories', 'feature_names', 'target_names', 'DESCR', 'details', 'url'])
```
* **It'll take a long time**

This will return a dictionary with various information about the dataset, including the data and target values separated. 
* *data* and *target* are pandas dataframes

# 🔵 MNIST
This is the dataset that contains images of handwritten digits, used for classification problems.

The target values in this dataset come as strings, so it is important to convert them:
```python
y = y.astype(np.uint8)
```

## 🔷 Printing Data as Images 
The images in the dataset are given with 784 features, each feature represents a pixel, and its value can range from 0 (white) to 255 (black).

To plot them, we must first *reshape* them to a 28x28 array, and then pass it to matplotlib:

```python
import matplotlib as mpl
import matplotlib.pyplot as plt

def show_image(data, digit):
    some_digit = np.array(data.iloc[digit])
    some_digit_image = some_digit.reshape(28, 28)
    
    plt.imshow(some_digit_image, cmap = mpl.cm.binary)
    plt.axis("off")
    plt.show()

show_image(X, 0)
```

## 🔷 Binary Classifier
Let's create a 5-detector (binary classifier) using `SGD` - Stochastic Gradient Descent
```python
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train.values, y_train_5.values)
```
