### ❗ What do To?
When the error is not satisfactory (or just insanely good) you must first try to understand what could have happened:

1. **Overfitting:** You got an unexpectedly good error (equal or close to 0).
	* You could try using a less powerful model
	* Tweak the hyperparameters
	* Add more data

2. **Underfitting:** The error is just plain bad - too big for what you need. 
	* You could try using a more powerful model
	* Add more data
	* Add more features (the current ones doesn't provide enough information)
	* Better prepare your features (they could have low quality)
	* Reduce regularization of your model

Usually try to do what is easier first - and train new models is very easy and fast, so it is a good starting point. 