A machine learning model is a program that is used to make predictions for a given data set. A machine learning model is built by a supervised machine learning algorithm and uses computational methods to “learn” information directly from data without relying on a predetermined equation. More specifically, the algorithm takes a known set of input data and known responses to the data (output) and trains the machine learning model to generate reasonable predictions for the response to new data.

Table of Contents

Types of Machine Learning Models
Machine Learning Models with MATLAB
Integrate Machine Learning Models into Systems
Types of Machine Learning Models
There are two main types of machine learning models: machine learning classification (where the response belongs to a set of classes) and machine learning regression (where the response is continuous).

Choosing the right machine learning model can seem overwhelming—there are dozens of classification and regression models, and each takes a different approach to learning. This process requires evaluating tradeoffs, such as model speed, accuracy, interpretability, and complexity, and can involve trial and error.

The following is an overview of machine learning classification and regression machine learning models to help you get started.

Diagram of supervised learning divided into classification and regression with several machine learning models under both categories.
Machine learning models for classification and regression with MATLAB.

Popular Machine Learning Models for Classification or Regression
Machine Learning Model

How Machine Learning Model Works

Machine Learning Model Representation

Support Vector Machine (SVM)

An SVM classifies data by finding the linear decision boundary (hyperplane) that separates data points of one class from data points of the other class. The best hyperplane for an SVM has the largest margin between the two classes when the data is linearly separable. If the data is not linearly separable, a loss function penalizes points on the wrong side of the hyperplane. SVMs sometimes use a kernel transformation to project nonlinearly separable data into higher dimensions where a linear decision boundary can be found.

SVM regression algorithms work like SVM classification algorithms, but the regression algorithms are modified to predict continuous responses. They find a model that deviates from the measured data with minimal parameter values to reduce sensitivity to errors.

Classification

SVM model

Regression

SVM Regression model

Decision Tree

A decision tree lets you predict responses to data by following the decisions in the tree from the root (beginning) down to a leaf node. A tree consists of branching conditions where the value of a predictor is compared to a trained weight. The number of branches and the values of weights are determined in the training process. Additional modification, or pruning, may be used to simplify the model.

Regression tree model

Ensemble Trees

In ensemble methods, several weaker decision trees are combined into a stronger ensemble. A bagged decision tree consists of trees that are trained independently on bootstrap samples of the input data.

Boosting involves iteratively adding and adjusting the weight of weak learners. There is an emphasis on misclassified observations or fitting new learners to minimize the mean-squared error between the observed response and the aggregated prediction of all previously grown learners.

Regression tree ensembles model

Generalized Additive Model (GAM)

GAM models explain class scores or response variables using a sum of univariate and bivariate shape functions of predictors. These models use a shape function, such as a boosted tree, for each predictor and, optionally, each pair of predictors. The shape function can capture a nonlinear relation between predictors and predictions.

gam

Neural Network

Inspired by the human brain, a neural network consists of interconnected nodes or neurons in a layered structured that relate the inputs to the desired outputs. The machine learning model is trained by iteratively modifying the strengths of the connections so that given inputs map to the correct response.

The neurons between the input and output layers of a neural network are referred to as hidden layers. Shallow neural networks typically have few hidden layers. Deep neural networks have more hidden layers than shallow neural networks. They can have hundreds of hidden layers.

Neural networks can be configured to solve classification or regression problems by placing a classification or regression output layer at the end of the network. For deep learning tasks, such as image recognition, you can use pretrained deep learning models. Common types of deep neural network are CNNs and RNNs.

Deep Neural Network model

Popular Machine Learning Models for Classification
Machine Learning Model

How Machine Learning Model Works

Machine Learning Model Representation

Naive Bayes

A naive Bayes classifier assumes that the presence of a feature in a class is independent of other features. It classifies new data based on the highest probability of it belonging to a particular class. This probability is determined by the probabilities of each feature.

Naive Bayes model

k-Nearest Neighbor (KNN)

KNN is a type of machine learning model that categorizes objects based on the classes of their nearest neighbors in the data set. KNN predictions assume that objects near each other are similar. Distance metrics, such as Euclidean, city block, cosine, and Chebyshev, are used to find the nearest neighbor.

KNN model

Discriminant Analysis

Discriminant analysis classifies data by finding linear combinations of features. The analysis assumes that different classes generate data based on Gaussian distributions. Training the machine learning model involves finding the parameters of a Gaussian distribution for each class. Boundaries, either linear or quadratic, are calculated based on these parameters. These boundaries are used to classify new data.

Discriminant Analysis Ensembles model

Popular Machine Learning Models for Regression
Machine Learning Model

How Machine Learning Model Works

Machine Learning Model Representation

Linear Regression

Linear regression is a statistical modeling technique used to describe a continuous response variable as a linear function of one or more predictor variables. Because linear regression models are simple to interpret and easy to train, they are often the first models to try when working with a new data set.

Linear regression model

Nonlinear Regression

Nonlinear regression is a statistical modeling technique that helps describe nonlinear relationships in experimental data. These models are generally assumed to be parametric, described as nonlinear equations.

Nonlinear refers to a fit function that is a nonlinear function of the parameters. For example, if the fitting parameters are b0, b1, and b2: the equation y = b0+b1x+b2x2 is a linear function of the fitting parameters, whereas y = (b0xb1)/(x+b2) is a nonlinear function of the fitting parameters.

Nonlinear regression model

Generalized Linear Model

A generalized linear model (GLM) is a special case of nonlinear models that uses linear methods. The inputs are transformed by a nonlinear link function such as a logarithm or logit function. The linear combination of transformed inputs is solved using a linear best fit. The logistic regression machine learning model is an example of a GLM.

Generalized linear model

Gaussian Process Regression (GPR)

GPR models are nonparametric machine learning models used for predicting the value of a continuous response variable. The response variable is modeled as a random Gaussian process, using covariances with each input variable. The machine learning model also models the uncertainty of the response.

These models are widely used in spatial analysis for interpolation in the presence of uncertainty. GPR is also referred to as Kriging.

GAM model

Machine Learning Models with MATLAB
Using MATLAB® with Statistics and Machine Learning Toolbox™, you can train many types of machine learning models for classification and regression. The following tables list MATLAB functions that create popular machine learning models and documentation topics, which describe how the machine learning models work. With MATLAB, you can create more machine learning models than the ones listed here.

Machine Learning Models for Classification
Machine Learning Model

MATLAB Function that Creates Machine Learning Model

How Machine Learning Model Works in MATLAB

SVM

fitcsvm	
 

Naive Bayes

fitcnb	
 

KNN

fitcknn	
 

Decision Tree

fitctree	
 

Decision Tree Ensemble

fitcensemble	
 

Discriminant Analysis

fitcdiscr	
 

GAM

fitcgam	
 

Shallow Neural Network

fitcnet	
 

Deep Neural Network

trainNetwork	
 

Machine Learning Models for Regression

Machine Learning Model	
MATLAB Function that Creates Machine Learning Model

How Machine Learning Model Works in MATLAB

Linear Regression

fitlm	
 

Nonlinear Regression

fitnlm	
 

GPR

fitrgp	
 

SVM

fitrsvm	
 

Generalized Linear Model

fitglm	
 

Tree

fitrtree	
 

Regression Tree Ensemble

fitrensemble	
 

GAM

fitrgam	
 

Shallow Neural Network

fitrnet	
 

Deep Neural Network

trainNetwork	
 

MATLAB provides low-code apps (Classification Learner (4:34) and Regression Learner (3:42)) for designing, tuning, assessing, and optimizing machine learning models.

Classification Learner App
Screenshot of a classification machine learning model visualized with the Classification Learner app.
Regression Learner App
Screenshot of a regression machine learning model visualized with the Regression Learner app.
Specialized apps for interactively exploring data, selecting features, and training, comparing, and assessing machine learning models.

With MATLAB, you can automate the process of building optimized machine learning models. Using autoML techniques, you can streamline data exploration and preprocessing, feature extraction and selection, machine learning model selection and tuning, and the preparation of a machine learning model for deployment.

Integrate Machine Learning Models into System
Using MATLAB and Simulink® with Statistics and Machine Learning Toolbox, you can integrate machine learning models into the design and simulation of complex AI-driven systems, such as a reduced-order model of a vehicle engine.

