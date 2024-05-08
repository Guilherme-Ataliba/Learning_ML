# 🔵 Understanding - 1º Way

You can think of an array as a collection of arrays of 1 dimension lower than the array's dimension. Meaning that:

* **Vector:** An array of numbers (0-d array)
```python
vector = np.array([0, 1, 2])
```

* **Matrix:** An array of vectors (1-d array)
```python
matrix = np.array([
				   [0, 1, 2], 
				   [3, 4, 5]
				   ])
```

* **3-array**: An array of matrices (2-d array)
	* 3-arrays can be seen as a block of matrices (tensor)
```python
array_3 = np.array([
					[[0, 1, 2], 
					[3, 4, 5]],
					
					[[6, 7, 8], 
					[9, 10, 11]]
					])
``` 
![[Pasted image 20230824235929.png]]

* **4-array**: An array of 3-arrays (3-d array)


Think in indices, how do you select an element in each type? For example, in a 3-array:
```python
array_3[0]
```
* It'll return a matrix (same for `array_3[1]`)

The main point is, a 3-array is a collection of matrices, the first index (axis) chooses a matrix in the collection. 
* Furthermore, a 4-array is a collection of 3-arrays, meaning that, the first index will choose a block of matrices.


# 🔵 Understanding - 2º Way
You can visualize an array as a structure where each element of the array is an array of one lower dimension. Meaning that, high dimensional arrays can be expressed as **blocks**. 

* **Vector**: An array of 0-arrays (numbers)
* **Matrix**: It is a collection of arrays
* **3-array**: Is a matrix where each element of it is an array itself
```python
# 3-array
| [1, 2]    [3, 4]    [5, 6]   | 
| [7, 8]    [9, 10]   [11, 12] |
| [13, 14]  [15, 16]  [17, 18] |
```
**4-array**: And just like that, a 4-array can be seen as a matrix where each element of it is a matrix itself
```python

# this can be seen as a 2x2 matrix where each element is a 2x2 matrix
| | [1, 2]  [3, 4] |     | [1, 2]  [3, 4] | |
| | [5, 6]  [7, 8] |     | [5, 6]  [7, 8] | |
|                                           |
| | [1, 2]  [3, 4] |     | [1, 2]  [3, 4] | |
| | [5, 6]  [7, 8] |     | [5, 6]  [7, 8] | |
```


# 🔵 Axis
Is a way of expressing what each indexing of an array should return - for example, the first index of a 3-array return a matrix, the first index is the axis zero.
![[Pasted image 20230824235929.png]]![[Pasted image 20230825001227.png]]

Axis are counted from outside-in, meaning that, the highest dimension comes first, and it encompasses the biggest cluster of arrays. 


# 🔵 Transpose
Transposing an n-array consists of switching two axes. 
* For a matrix, you swap a column and row
* For an n-array you swap two axes.

### 🔷 Knowing How The Transpose Will Change
Let's say you have a 3-array and want to make a transpose of it. 
```python
>>> array([[[ 1,  2,  3,  4],
        [ 5,  6,  7,  8],
        [ 9, 10, 11, 12]],

       [[13, 14, 15, 16],
        [17, 18, 19, 20],
        [21, 22, 23, 24]]])
```

*What will happen if you swap the axis 0 and 1?* 
1. Know what each axis (index) represents
	* 0 - which matrix
	* 1 - rows inside matrix

Think in layers, a 4-array is just a matrix where the elements are matrices. So, if you want to:
2. Change the two out-most axis (i.e. 0, 1), you can just look at it as a regular matrix transpose!

Now, if you look at a 3-array and want to 
3. Swap an inside and an outside axis (for example, 0 and 2), the inside axis will stay in the **same order** and with the **same size**, but it'll jump to the next dimension every time it gets to the end

```python
>>> A = array([[[ 1,  2,  3,  4],
        [ 5,  6,  7,  8],
        [ 9, 10, 11, 12]],

       [[13, 14, 15, 16],
        [17, 18, 19, 20],
        [21, 22, 23, 24]]])
```

```python
A.transpose(0, 1, 2)
A

>>>  array([[[ 1, 13],
        [ 5, 17],
        [ 9, 21]],

       [[ 2, 14],
        [ 6, 18],
        [10, 22]],

       [[ 3, 15],
        [ 7, 19],
        [11, 23]],

       [[ 4, 16],
        [ 8, 20],
        [12, 24]]])
```
* Notice how we swapped the third dimension with the column dimension, and the columns were arranged vertically into matrices - jumping to the next matrix

> You had \[1, 5, 9\] in the 1º column of the 1º matrix and \[13, 17, 21\] in the 1º column of the 2º matrix - after the transpose the 1º matrix is 
> \[\[1, 5, 9\], \[13, 17, 21\]\] 



# 🔵 Reshape
Changes the shape of an n-array. The shape you want to cast into must have the same number of elements the array currently has. 

Reshape, works by changing the shape of the elements in the order you'd usually count an array / matrix.