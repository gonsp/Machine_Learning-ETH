# Approaching ML Projects
The purpose of this tutorial is to give broad tips for solving the types of problems encountered in the ML projects. This is not to indicate that these tips are magic tricks for solving any particular project, but rather they are suggestions for _how_ to approach these problems.

## 0. Meta-remarks: The Setting
The ML projects put you in the role of an applied machine learner who has been given data and a corresponding task to solve on it. Since you have relatively little opportunity to interact with the task-setting/data-providing collaborator, many of these tips revolve around the need to make sensible choices in the absence of domain-specific feedback from the collaborator. In ideal scenarios, you will be able to ask arbitrary questions of this collaborator. However, this will often not be the case, and you need to be able to make reasonable decisions about tackling the task even without full information about the problem.

The ML projects are also special in that they define, from the outset, a problem of interest with corresponding labels. In real situations, identifying interesting prediction problems and defining appropriate labels can be a nontrivial collaborative task between machine learners and scientists (or other data-holders).

## 1. What's the problem?
### Label type
Broadly speaking, most machine learning problems are either _regression_ or _classification_. You should confirm that the labels available to you make sense given the description of the task.

> **When classification looks like regression**
>  A rule of thumb for distinguishing classification and regression is that regression has _numerical_ labels, whereas classification has _categorical_ labels. However, oftentimes when we perform classification, we actually output the _probability_ of an example belonging to a particular class - this probability is a real number, and it's what enables us to use regression-like techniques (such as logistic regression) to solve classification problems.

Understanding your labels allows you to choose the right class of models for the problem. If your labels take values between 0 and 1, there's no point having a model which can output negative values.

### Look at the labels
Look at the prevalence of the different types of labels. For classification, how many examples of each class do you have? For regression, what does the distribution of labels look like? If your data is very _imbalanced_, you may need to account for this during training and evaluation. [Here](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/) is one resource for dealing with class imbalance.

## 2. Know your data

Knowing a little bit about your data is useful for multiple reasons:

 - Facilitating communication with collaborators (not applicable for the ML projects, but broadly important to remember)
 - Making reasonable modelling assumptions
 - Making reasonable preprocessing decisions, including feature engineering

> By **modelling assumptions** I mean the assumptions you make about the data which you encode in your choice of model. Linear models assume _independent, additive_ effects of covariates. Convolutional neural networks (generally) assume _local structure_. For probabilistic models, these assumptions are typically about which probability distribution describes the data.

It is good practice to check your assumptions. Data you assume to be normally-distributed may not be. Data you assume to be regularly-sampled may not be. Data you assume to be free of artefacts probably isn't. It never hurts to _visualise your data_.

**Preprocessing** can be roughly split into two tasks:

### 2.1. Data quality control

For the ML projects, you can assume Point 1 has largely been taken care of. This task consists of some or all of the process of taking the data from its 'raw', freshly-collected state, to something appropriate to give to a Kaggle competition. A related, somewhat more general article: [Data Readiness Levels: Turning Data from Palid to Vivid](http://inverseprobability.com/2017/01/12/data-readiness-levels).

### 2.2. Preparing data for use in a model

This point, although broadly about feature engineering, is model dependent. Some models (random forests are a notable example) require relatively little data preparation. For the projects, you will probably spend some time thinking about feature engineering.

#### Feature Engineering

Feature engineering operates in conjunction with modelling assumptions. A model which is not linear in _x_ and _y_ may be linear in _xy_, as a simple example. Knowing something about the data and the task can be crucial at this stage.

Here are some approaches you can take to determine useful features:

  - **Visualisation**: If there are any univariate associations between your features and the labels, simple scatter plots or histograms (if your features are categorical) will yield them. The relationship is almost certainly not a univariate one, but this simple exercise can nonetheless give valuable insight into the data and suggest further hypotheses to visually test.
   - For visualising high-dimensional data, there are dimensionality reduction techniques such as principal component analysis, clustering, and [t-SNE](https://distill.pub/2016/misread-tsne/) which may yield some insight. Such lower-dimensional representations of the data can also be used as input features themselves.
  - **Domain knowledge**: What is already known to be useful for processing this type of data? You may not be an expert in (for example) ECG time-series analysis, but research into standard practices can yield possible features of interest. There may even be machine learning papers describing tasks on similar datasets.
   - A note on *time series*: there is an enormous body of knowledge on the topic of processing signals in the form of time series, in the field of signal processing. [scipy](https://docs.scipy.org/doc/scipy/reference/signal.html) supports many relevant functions. Be aware that not every signal is created equal, and different fields may have their own approaches for processing time series.
  - **Trial and error**: Yes, seriously. This is usually called 'grid search' or 'manual hyperparameter optimisation' to mask the grim truth. This isn't as trivial as it sounds, because running trial-and-error experiments requires some set-up, and it also raises the risk of overfitting (as any optimisation procedure can). Using the data you have, you should create internal validation sets (as in cross-validation) to help in this task. You shouldn't use the test set performance to select your choice of features because:
  1. Fitting your model (in any way) to the test set destroys your ability to evaluate how well it performs on _completely unseen_ data, making it impossible to evaluate. It also destroys your ability to make it past peer review, for good reason.
  2. In Kaggle-style competitions, your access to the test set is usually limited, so even if you _wanted_ to use the test set, you practically can't ([unless you do what these researchers got in a lot of trouble for](https://www.nytimes.com/2015/06/04/technology/computer-scientists-are-astir-after-baidu-team-is-barred-from-ai-competition.html) - don't do this).

Another aspect of feature engineering is feature _selection_. In non-regularised models, using too many features introduces too many free parameters, and can result in rapid overfitting. You can avoid this by (surprisingly enough) using regularised models (e.g. with LASSO), or by first performing a feature selection procedure. [Here](http://blog.datadive.net/selecting-good-features-part-i-univariate-selection/) is an informative blog post about univariate feature selection.

#### Avoiding Manual Feature Engineering

Simple models often need complicated features. Complicated models can sometimes get away with simple features. An example you are familiar with from class are **kernel-based** models such as support vector machines. In that case, the feature engineering task can amount to simply choosing a kernel.

Another famous class of complicated models are deep neural networks.

> One of the strengths of **deep learning** is "automatic feature extraction". The idea is that a sufficiently expressive network can take mostly-unprocessed inputs and perform transformations to extract the most important features for the task.

It is important to mention that deep models (indeed, any models optimised using gradient descent) perform best when their inputs are **normalised**. Technically this can refer to transforming the data to follow a standard normal distribution, but in practice it amounts to subtracting the mean of the data and dividing by the standard deviation (done for each feature independently). [Here](https://www.coursera.org/learn/deep-neural-network/lecture/lXv6U/normalizing-inputs) is a short video about the importance of normalising features for deep networks.

An important caveat of this normalisation procedure is that it works best on data that's already roughly normally distributed. If you blindly apply this to binary-valued data, you may introduce highly _non_-normal values. (Consider what happens if 99% of the values are 0.)

## 3. Choosing a model
As evidenced by the previous section, selecting a model class goes hand in hand with thinking about preprocessing and preparing the data. Luckily, people have made 'cheat-sheets' to aid in the process of choosing the right model.

 - [Choosing the right estimator](http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html) from scikit-learn
 - [Machine learning algorithm cheat sheet](https://docs.microsoft.com/en-us/azure/machine-learning/studio/algorithm-cheat-sheet) from Microsoft
 - [Another cheat sheet](https://blogs.sas.com/content/subconsciousmusings/2017/04/12/machine-learning-algorithm-use/) from SAS

Here's my advice: 

  - Start simple, then add complexity.  Don't start implementing a deep neural network before you've tried logistic regression. If the data is really big, don't spend hours fitting a model before you're tried it on a (representative) subsample.
  - Use existing resources (like scikit-learn and other ML toolboxes) to save yourself time both implementing and debugging models. Re-inventing the wheel is a time-consuming if educational exercise.


## 4. Final suggestions

### 4.1 Sanity checks

It's very easy to write machine learning code which runs, producing output of the expected type, raising no (python) errors, and which is _completely useless_. 

Semantic errors can creep in very easily. These are errors like using the training set instead of the test set ("wow, my accuracy is 99.9%!"), shuffling the features but not the labels ("why is my model performing no better than random?"), forgetting to increment a counter while processing data ("why are most of the values 0?"), and just about every other silly mistake you didn't anticipate which can render your model useless and your score inadequate.

One method to prevent this is to perform lots of _sanity checks_.  Visualisation is your friend. Assert statements are your friend.

Some example sanity checks:

  - Did you implement a fancy new model but it's not working? Check if it works on a _really easy_ task (you can generate synthetic data for this) - if it still doesn't work, you probably have a bug, or your model has some deeper issues!
   - If you're optimising it with stochastic gradient descent (as you likely would for a neural network), plot the training set loss and validation set loss during training to keep an eye on it. If the training set loss is going down, but the validation set loss is barely moving, your model is 'working', it's just not _generalising_.
  - Is your model working _suspiciously well_? See what happens when you give it random noise instead of its real training labels - it should perform like random on the validation set, and if it doesn't... something is up. You can also make the task 'harder' for it by giving it only a fraction of the training data or features.
  - Stumped by why your model isn't generalising? Maybe you forgot to process the validation/test set in the same way as the training data - plot the distribution of features in both training set and test/validation set to check this. You might find something weird about your train/validation/test split along the way, too.

These are fairly high level sanity checks. On a lower level, you can include checks in your code, to make sure e.g. the features and labels have the same number of examples, to make sure no values are below 0 (if that's a property of your data or labels), to make sure things which should be probabilities add up to 1. Many of these types of checks are included in scikit-learn models, for example, but it never hurts to ensure that your beliefs correspond to reality.

### 4.2 Learn from others

Since the ML projects take the format of a Kaggle competition, experiences from Kagglers is arguably relevant, and may provide additional angles and suggestions not covered in this tutorial. Some resources:
  
   - [Here's a video from the CEO of Kaggle](https://www.youtube.com/watch?v=GTs5ZQ6XwUM&feature=youtu.be) on what people do in Kaggle competitions. 
   - Here's a blog post from a Kaggler: [How to Rank 10% in Your First Kaggle Competition](https://dnc1994.com/2016/05/rank-10-percent-in-first-kaggle-competition-en/) (he mentions the utility of **ensemble models**)
   - A slide deck - [Tips and tricks to win kaggle data science competitions](https://www.slideshare.net/DariusBaruauskas/tips-and-tricks-to-win-kaggle-data-science-competitions)
   - Another series of blog posts: [How Feature Engineering can help you do well in a Kaggle competition - Part I](https://medium.com/unstructured/how-feature-engineering-can-help-you-do-well-in-a-kaggle-competition-part-i-9cc9a883514d)

More generally: data science and machine learning are rapidly expanding fields, with multitudes of questions on StackExchange and other sites, and informative blog posts and articles (some of which I've included here).

And last but not least: there's always Piazza.