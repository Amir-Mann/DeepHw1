r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**

**1. False  2. False  3. True  4. True**

**explanation**

1. The in-sample error is a deterministic value calculated on the training dataset (only), 
   independed of the test dataset.
   The test dataset helps us to estimate the out-of-sample error.

2. We can split out dataset into a zero size training dataset, and full size test dataset.
   such split would make any model equally good. and so training would not change our initial guess.
   and no learning would be done at all.
   
3. In cross validation we change and train our training and validation split to get a better optimization for our hyperparameters.
   using the test set in our cross validation would contaminate our out-of-sample estimation. so we should not use it.

4. Generalization error is the error of the model on unseen data.
   In cross validation we set a side some of the data (different group each fold) and not
   expose it to the model while training, and then that data is used to evaluate the generalization error.
"""

part1_q2 = r"""
**Your answer: Not justified**

By using the test set to determine the best $\lambda$ we damage the ability to estimate our model's out-of-sample error.
Because we choose the $\lambda$ that will produce better results specificly on the test set and therefor the test set will no longer
quilify as unseen data new samples from the theoretical distribution.
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**

Increasing K may lead to improved generalization of the model, up to a certain point.
When $K=1$ the model is overfitted, the training error would be 0, but every prediction on
unseen data would be extremly sensitive to it's nearst neighboor, making for bad generalization.
When $K>=N$, where $N$ is the trainig data size, we get a simple decision rule predicting each sample
to be the common label in the data, making for very bad "model".
So there should be an increase in K which will improve model preformance on unseen data, but
only up to a certain point which is depandent on the distribution and number of samples.
"""

part2_q2 = r"""
**Your answer:**

Training with k-fold CV is better-

1. than finding best model with regard to *train-set* acuraccy because it minimazes the in-sample loss,
   which in itself has no value to the actuall accuracy on unseen data which is our actuall goal.
   This will make for an overfitting of the model structure itself to data. (on KNN it will always set k=1)
2. than finding best model with regard to *test-set* acuraccy because it changes our model directy for
   fitting the test data, destroying our ability to estimate the out-of-sample error once we finished
   training our model.
"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**

1. the model is a simple linear classifier. as so, it learns to predict the digit according to 
   activated pixels in the image. most of the wrong prediction of our model were "false 8",
   means that the model mistakenly classified different digits as 8.
   in those cases, we noticed that the center of the image consists large amout of activated pixels,
   so as our model is a simple linear classifier it makes sense that it would be wrong
   in cases like described above.

2. this interpretation differs from KNN, because it looks on behaviors of digits. for example, all 7's has
   a black section in the right bottom corner. where as KNN looks on the K-nearest neighbors. for example for k = 3, 
   it would'nt so heaviley mispredict 3's as an 8 because there will be enough 3's that are highly activated 
   in the center.  


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**
The learning rate we choose is good, maybe a little fast.
We can see that on the last few epochs the model did not improve on the validation set at all.
If we would have choosen a too high learning rate the model would have picked much earlier in the 
learning proccess and would'nt have stayed stable after that.
If we would have choosen a too low learning rate the model wouldn't have reached 0 incline, and
would have required more epochs to finish the learning.

Based on the graph we can say that the model learned pretty well, and is generalzing pretty well.
We might say that this model is slightly underfitted - even on the training dataset the model
doesn't reach solid 92+% accuracy at any given point, this is duo to the simplicty of the linear
clasiffier.
"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**

The ideal pattern would have been a $e = 0$ line.
We can say that the model is fairly descent at evaluating data, and that the CV did 
infact make for good generalzing learning model, because the train errors are similar
to the test errors.
The final plot after CV is much more dense around $e = 0$ line, it is also clear that in the
top 5 model alot of prediction prduced big negative residules, and the final model is better
balanced.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q2 = r"""
**Your answer:**

1. it is still a linear regression model. that because our model still performs linear interpretation of the
   features it's given, simply the features have been tranformed not linearily. 
2. yes we can. because bob the builder said so, and also because every function
   $f: \mathbbR^d \rightarrow \mathbbR$ represanting the perfect nonlinear function, by using the transformation
   $x' = f(\bar x)$ would allow for a linear model to predict f perfectly.
3. No, it would'nt be a hyperplane anymore. although it would be a hyperplane in the transformd features space, 
   when reducing the hyperplane to a surface in the original feature space, we get a complex nonlinear surface.
   the nonlinearity in the features themselves, will make for a curved decision boundary.

"""

part4_q3 = r"""
**Your answer:**

1. We used np.logspace because the regularization is much more sensevite to values around 0 than around 100.
2. $folds ampunt * lambdas amount * degrees * cols_numd amount$, which equals to $3 * 20 * 4 * 10 = 2400$ 

"""
# ==============
