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
as the same way as the majority of the data, making for very bad "model".
So there should be an increase in K which will improve model preformance on unseen data, but
only up to a certain point which is depandent on the distribution and number of samples.
"""

part2_q2 = r"""
**Your answer:**

Training with k-fold CV is better-

1. than fining best model with regard to *train-set* acuraccy because it minimazes the in-sample loss,
   which in itself has no value to the actuall accuracy on unseen data which is our actuall goal.
   This will make for an overfitting of the model structure itself to data.
2. than fining best model with regard to *test-set* acuraccy because it changes our model directy for
   fitting the test data, destroying our ability to estimate the out-of-sample error once we finished
   training our model.
"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


The choice for $\Delta$ is arbitrary because it makes for the same loss function multiplied by a constant, and proportional weights which do not change the final optimized decision boundary.
Lets mark the original parameters $L, W, \lambda, \Delta$ and the new ones $L', W', \lambda', \Delta'$, lets then assume $\Delta' = C\cdot\Delta$ for some $C\in R$.
For $W' = C\cdot W, \lambda' = \frac{\lambda}{C}$
We get:
$$ L'(W) = \frac{1}{N}\sum_i\sum_{j\neq y_i} max(0, \Delta' + w'^T_j \cdot x_i - w'^T_{y_i} \cdot x_i) + \frac{\lambda'}{2}||W'||^2 $$
$$
 = \frac{1}{N} \sum_i \sum_{j \neq y_i} max(C \cdot 0, C \Delta + C w^T_j \cdot x_i - C w^T_{y_i} \cdot x_i) + \frac{\lambda}{2C}||C \cdot W||^2$$
 $$
 = C \frac{1}{N} \sum_i \sum_{j \neq y_i} max(0, \Delta + w^T_j \cdot x_i - w^T_{y_i} \cdot x_i) + \frac{\lambda C^2}{2C}||W||^2$$
$$
 = C L(W)$$

 There for the optimization and optimal solution would be parallel to the original one and the classification would yield the same result (up to changes in the initial parameters, like $\lambda$ and $W_0$.

"""

part3_q2 = r"""
**Your answer:**

1. the model is a simple linear classifier. as so, it learns to predict the digit according to 
   activated pixels in the image. we noticed, that our model mis-classified digits (for example
   mistakenly classified sample to be the digit y) for samples that were activated in parts that 
   were characterized with the digit y. 
   means that if a sample of the digit 4 were highly activated strongly in the right half, and at
   the general top area, our model might have classified this sample as 9, as digit that is activated
   in the same regions.

2. this interpretation differs from KNN, because it looks on behaviors of digits. for example, all 7's has
   a black section in the right bottom corner. where as KNN looks on the K-nearest neighbors. for example for k = 3, 
   it would'nt so heaviley mispredict 3's as an 8 because there will be enough 3's that are highly activated 
   in the center.  

"""

part3_q3 = r"""
**Your answer:**
The learning rate we choose is good, maybe a little fast.
We can see that on the last few epochs the model did not improve on the validation set at all.
If we would have choosen a too high learning rate the model would have picked much earlier in the 
learning proccess and would'nt have stayed stable after that.
If we would have choosen a too low learning rate the model wouldn't have reached 0 incline, and
would have required more epochs to finish the learning.

Based on the graph we can say that the model learned pretty well, and is generalizing pretty well.
We might say that this model is slightly underfitted - even on the training dataset the model
doesn't reach solid 92+% accuracy at any given point, this is due to the simplicty of the linear
clasiffier.
"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**

The ideal pattern would have been an $e = 0$ line.
We can say that the model is fairly descent at evaluating data, and that the CV did 
infact make for good generalizing learning model, because the train errors are similar
to the test errors.
The final plot after CV is much more dense around $e = 0$ line, it is also clear that in the
top 5 model alot of prediction prduced big negative residules, and the final model is better
balanced.

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
