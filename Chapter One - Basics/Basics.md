
# 🔵 Machine Learning - ML

Definition:
> A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E

* A performance measure, an error, is essential for any machine learning algorithm


### 🔹 Definitions
* The examples that the system uses to learn are called the **training set**
	* Each training example is called a training instance (or **sample**)
	* **Feature**: Is a data type, a characteristic measured and used to train the algorithm
	* **Label**: Is the *response / solution* expected to a given set of features. This is used in supervised learning, where the label is the expected value. 


## 🔷 Types of Problems in ML

🔹 Supervised Learning                   🔸 Unsupervised Learning

### 🔹 Classification
As the name suggests, is the problem of classifying a new entry into one of the existing classes. In statistical language, you have a partitioned sample space and, you want to classify your event in one of the partitions.

### 🔹 Regression
Problems are those in which you want to *predict* a target numerical value, given a set of features. The algorithm will use regression to build a function that predicts new entries of features and outputs a prediction. 

### 🔸 Clustering 
The algorithm cluster the data together, not necessarily into a class since it has not yet been established based on similar features between the data. Then you can use that ML algorithm to classify new data into those clusters. 

### 🔸 Anomaly Detection
Is the problem of detecting deviations from the expected behavior withing the data. The algorithm does this by comparing all data and analyzing which ones deviate from the rest. (kind of like clustering). 

Instead of needing data from failures, to be able to detected one, we just need data from the normal behavior, and then we detected a deviation from that normal behavior.

It can be used in:
* Financial fraud 
* Manufacturing defects
* Unusual movement in surveillance footage. 
* **Remove outliers** 

### 🔸 Novelty Detection / One-Class Classifier 
You have a set of known data and their classifications. Novelty detection algorithms face the problem of, when a new entry is exposed to the old data, to classify whether the new data fits in the old data or is new compared to what is known, that is, if the new data is a novelty or not (compared to the known classifications).

The set of "known" data can be multi-classified, we just draw an out-line around what we know and what is considered novelty.

**Anomaly Detection X Novelty Detection**
* The difference from anomaly detection algorithms is that novelty detection can only be trained in normal data, whilst anomaly detection algorithms tolerate some outliers in the training set. 

Ex: If you collect data from a volcano and want to predict when it's going to erupt, you need to detected "novelty data", since all you have collected is "normal data" and you must be able to detected when the collected data differ from the normal - much like anomaly detection 

### 🔸 Dimensionality Reduction
As the name suggests, it's the problem of reducing the number of dimensions (features) required to describe a problem, without altering too much of the results. 

### 🔸 Visualization
Algorithms take your high dimensional data and output a 2D or 3D plot, for visualization purposes. They don't actually reduce the data dimension's, just find a way to plot it in 2D or 3D. 

### 🔸 Association Rule Learning
Given some data class (suppose that has happened), what are the chances that another class happens - this is the problem of associating different classes together. It's the problem of digging into large amounts of data and finding interesting relations between attributes.

This can be used to recommend similar items in stores, or put items that are usually bought by the same group together.


# 🔵 Types of Machine Learning


## 🔷 Supervised Learning
In supervised learning, the training data includes the features and the desired solutions to that set of features, called **labels**.

These labels are usually man-made, meaning that the data is collected from somewhere where a human response is used as a label. But this isn't always the case, since it's possible to use an ML algorithm to generate labels to another.

#### Algorithms
* K-Nearest Neighbors 
* Linear Regression
* Logistic Regression
* Support Vectors Machines (SVMs)
* Decision Trees and Random Forests
* Neural Networks


## 🔷 Unsupervised Learning
The training data is unlabeled, meaning the systems tries to learn with a teacher / without knowing what to expect. Some of the problems solved by this kind of ML are:

#### Clustering
* K-Means
* DBSCAN
* Hierarchical Cluster Analysis (HCA)

#### Anomaly Detection and Novelty Detection
* One-Class SVM
* Isolation Forest

#### Visualization and Dimensionality Reduction
* Principal Component Analysis (PCA)
* Kernel PCA
* Locally-Linear Embedding (LLE)
* t-distributed Stochastic Neighbor Embedding (t-SNE)

#### Association rule Learning
* Apriori
* Eclat


## 🔷 Semisupervised Learning 
These are algorithms that deal with partially labelled data, meaning most of the data is unlabeled and a little bit of it is labeled. 

Ex: Google Photos cluster the data based on a person. So it recognizes a person and then put all the photos that person in on in the same group. Then, the first time you label that person in a single photo, all photos are automatically labeled.

Semisupervised learning algorithms typically are a combination of supervised and unsupervised learning algorithms. 


## 🔷 Reinforcement Learning
The learning system, called **agent**, can observe, select and interact with the environment, and get rewards (or penalties - negative rewards). It must learn by itself what's the best strategy, called **policy**, to get the most reward over time. **Policy** is what defines which action the **agent** will take in some given situation.

--- 

# 🔵 Batch X Online Learning

## 🔷 Batch Learning / Offline Learning
The system is incapable of learning incrementally, that is, with data that is collected on the run of the algorithm. This kind of algorithm must be trained - **always** - on the whole available data, every time (even for updates).

This is why it's also called **offline learning**, since all the collected data will be brought offline, then you must retrain the algorithm, and finally re-upload it to collect more data, and then repeat the process. 
* This action can be easily automated, the only problem is the time it takes to train the algorithm on the whole data every time. 

## 🔷 Online Learning
You train the data incrementally by feeding data instances sequentially. Each learning step is cheap and fast, so the system can learn about new data on the fly, as it arrives. 

**Optimization:** Online learning algorithms are great if you have limited computing resources, since once an online learning system has learned about new data instances, you can discard them.
* It can also be used on HUGE sets of data, that wouldn't be possible to train on a single run. You divide them in smaller chunks and train each individually (even in multiple machines). 

**Learning Rate**: Represents how fast a system adapts to new data. It must be carefully optimized, since if the learning rate is too fast it'll quickly forget old data (for example, in a spam filter it'd only detect new kind of spams). If the learning rate is slower, it won't change very much, and it won't be very influenced by outliers. 

--- 

# 🔵 Instance-Based X Model-Based Learning


## 🔷 Instance-Based Learning
Are models that learn the training data by heart and apply some "proximity" metric on the new data to compare them with the known data. Then you can classify the new data as the classification of the closest one, or apply some calculations (like a medium value of the closest ones - **KNN**) to classify it. 

Instance based learning works by comparing new data to the known data. 


## 🔷 Model-Based Learning
These are models that develop a model (function) from the learning data and apply that model to classify/predict new data. Like in a regression problem, where you build a function, optimize the parameters and then use that function, optimized for the training set, to predict new entries. 


# 🔵 Model Tuning

## 🔷 Overfitting
It happens when the model performs well (usually very well) on the training set, but it does not generalize well. What happens is that the model "learns" the training set, it has learned to follow it too closely - thus it cannot generalize to new information.

A famous example is for a polynomial regression for a high degree polynomial, it passes on every point of the dataset, but it hasn't learned anything from the data, it just draws a line through every point. 
![[Pasted image 20230723093434.png]]

⭐ Overfitting happens when the model is too complex relative to the amount of noisiness of the training data. Some solutions are:

* To simplify the model by selecting one with fewer parameters (polynomial regression → linear regression) or by reducing the number of attributes in the training data
* To gather more training data
* To reduce the noise of the data (ex., fix the error and remove outliers)


## 🔷 Hyperparameters
These are parameters defined manually before the training processes, they aren't found by the algorithm, they're defined beforehand and can define what results/how the algorithm works. 

Ex:
* You can define the amount of layers in a neural network
* The learning rate in a logistic regression
* The number of trees in a random forest

These parameters define the algorithm, and it itself can't find them. It's the responsibility of the coder to find the best hyperparameters. This can be achieved in several ways:

* **Grid search**: Build a grid of possible combinations of hyperparameters, iterate over, retraining the model each time, them and find the combination the gives the best results
* **Random search**: Very similar to grid search, but instead of defining a grid, try a random combination of hyperparameters - this is usually computationally lighter 
* **Bayesian Optimization**: \*search\*


## 🔷 Regularization
Is the act of constraining a model to make it simpler and reduce the risk of overfitting. This means that, by limiting some parameters from the algorithm, you reduce its degrees of freedom, and thus make the algorithm simpler - by doing that you're trying to prevent overly complex algorithm that tend to overfitting. 

You still want to use that specific solution, but in a simpler manner - constrained for your purposes. The constraint applied usually is a **hyperparameter**, that must be tuned to achieve better results. 



# 🔵 Testing and validating
The best way to know how well a model will generalize to new cases is to actually test it on new cases. Given that metric, it is also important to tune what is possible in the algorithm, like the hyperparameters - to achieve a better result. To do all of this, there are some methods.


## 🔷 Train-Test Split
This is the basis of all methods of validation, it consists of splitting all the data into two parts:
* **Training Set**: Which is the data that you'll feed to the algorithm for training. This is where the most part of the data should be - since that's the part responsible for defining how good an algorithm is. 
* **Test Set**: This is the data that'll make predictions, using the trained model, to test the accuracy of the algorithm. You don't need a lot of data, just sufficient to make a good estimation of the model's precision. 

⭐ Usually the procedure divides the data into **80-20**, that is 80% of data the to the training set and 20% of the data to the test set. 


## 🔷 Model Selection
How do you select the best model for your specific problem? Well there isn't much alternative, you must train some models, tune them, and then find the best one to choose. After that, you can focus more on the chosen model and tune it more precisely.

## 🔷 Hyperparameter Tuning
Besides the parameter the algorithm itself finds, there are hyperparameters that must be defined beforehand and will greatly impact on the model's precision. There are ways to find the best hyperparameters that don't consist of manual search, like:

### 🔹 Search
You can define a grid of possible values to the hyperparameters, train the model for every hyperparameter combination and find the one that gives the best results. This can be done by:
* An actual **grid search**, where you define a grid of possible hyperparameters and iterate over them (possibly doing a distance parameter h, a start and end position)
* A **random search**, where the possible combinations are generated randomly, between a defined interval of interest. This is usually less computationally expensive. 
* Other methods...

#### ❗ Problem
The problem with a pure search is that you're prone to overfit the hyperparameters to the testing set, since you're trying to optimize them to this set - this won't probably generalize very well.


### 🔹 Holdout Validation 
To solve the problem of overfitting the hyperparameters to the testing set, we can use holdout validation. Which simply divides the training set into two parts

1. The first will be used to train and evaluate (tune their parameters) several candidate models and select the best one
2. **Validation Set**: The other part of the training set will be used to evaluate which is the best model - and tune their hyperparameters

After one model is chosen you must retrain it on the whole data - since it would be a waste of data otherwise - and finally test it to measure it's accuracy. 

🗒 k Fold Cross Validation