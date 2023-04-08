r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer: 1. False  2. False  3. True  4. True**

**explanation**
1.The in-sa0mple error a determnistic value calculated on the training dataset (only), 
  independed of the test dataset. 
  the test dataset helps us to estimate the out-of-sample error. 

2. We can split out dataset into a zero size training dataset, and full size test dataset.
   such split would make any model equally good. and so training would not change our initial guess.
   and no learning would be done at all.
   
3. In cross validation we change and train our training a and validation split to get a better optimization for our hyperparameters
   using the test set in our cross validation would contaminate our out-of-sample estimation. so we should not use it.

4. Generalization error is the error of the model on unseen data.
   In cross validation we set a side some of the data (different group each fold) and not
   expose it to the model while training, and then that data is used to evaluate the gen error.
"""

part1_q2 = r"""
**Your answer: Not justified**

by using the test set to determine the $\lambda$ we damage the ability to estimate our model's out-of-sample error.
because we choose the $\lambda$ that will produce better results specificly on the test set and therefoe the test seete will no longer
new unseen data from the theoretical distribution.
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


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
