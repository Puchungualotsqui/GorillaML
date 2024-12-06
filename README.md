# GorillazML

Gorillaz is a machine learning library written in Go. It provides tools for building and evaluating various machine learning models, including regression, classification, and utilities for model persistence and evaluation metrics.

![b3a37720-9ef2-4650-a9f3-d9ea04f74682](https://github.com/user-attachments/assets/1166e235-537a-48af-9d1d-16bd4d032dd3)


## Features

### Regression
- **Linear Regression**: Ordinary Least Squares, Ridge, Lasso, ElasticNet, Polynomial, Bayesian, and Robust regression models.
- **Customizable**: Offers tools for handling regularization and feature transformations.

### Classification
- **Decision Tree Classifier**: Highly configurable with parallel tree building and support for multiclass classification.
- **Linear Classifiers**: Logistic Regression and Support Vector Machines (SVM) with gradient-based training.
- **Naive Bayes**: Multinomial Naive Bayes for probabilistic classification.
- **k-Nearest Neighbors (KNN)**: KNN classifier with Euclidean distance for classification tasks.

### Utilities
- **Model Evaluation**: Includes metrics like accuracy, confusion matrix, precision, recall, and F1 score.
- **Model Persistence**: Save and load models using a simple API with Go's `gob` encoding.

---

## Installation

To use Gorillaz, install it using Go modules:

```
go get github.com/yourusername/gorillaz
```

## Example
```
package main

import (
	"fmt"
	"github.com/yourusername/gorillaz"
)

func main() {
	X := [][]float64{{1, 2}, {3, 4}, {5, 6}}
	Y := [][]float64{{1}, {2}, {3}}

	model := gorillaz.LinearRegression{}
	err := model.FitOLS(X, Y)
	if err != nil {
		fmt.Println("Error training model:", err)
		return
	}

	predictions, err := model.Predict([][]float64{{7, 8}})
	if err != nil {
		fmt.Println("Error making predictions:", err)
		return
	}

	fmt.Println("Predictions:", predictions)
}

```

## Regressions
### Linear Regressions
To perform any Linear Regression operation you need to use LinearRegression struct. It stores the coefficients and the Intercept points.
```
model := gorillaz.LinearRegression{}
```
#### FitOLS
Fit the model using Ordinary Least Squares regression.
- X *[][]float64*: independent variable data
- Y *[][]float64*: dependent variable data.
```
var err error
model := gorillaz.LinearRegression{}
err = mode.FitOLS(X, Y)
```
#### FitLasso
Fit the model using Lasso regression with L1 regularization.
- X *[][]float64*: independent variable data.
- Y *[][]float64*: dependent variable data.
- lambda *float64*: regularization strength; higher values enforce stronger sparsity.
- maxIter *float64*: maximum number of iterations for the coordinate descent optimization.
- tol *float64*: tolerance for stopping criteria.
```
var err error
lambda := 0.1     
maxIter := 1000
tol := 1e-6

model := gorillaz.LinearRegression{}
err = model.FitLasso(X, Y, lambda, maxIter, tol)
```
#### FitRidge
Fit the model using Ridge regression with L2 regularization.
- X *[][]float64*: independent variable data.
- Y *[][]float64*: dependent variable data.
- lambda *float64*: regularization strength; higher values apply stronger L2 penalties.
```
var err error
lambda := 0.5

model := gorillaz.LinearRegression{}
err = model.FitRidge(X, Y, lambda)
```
#### FitElasticNet
Fit the model using ElasticNet regression, which combines L1 and L2 regularization.
- X *[][]float64*: independent variable data.
- Y *[][]float64*: dependent variable data.
- lambda *float64*: overall regularization strength.
- alpha *float64*: balance between L1 (Lasso) and L2 (Ridge) penalties; alpha=1 is Lasso, alpha=0 is Ridge.
- maxIter *int*: maximum number of iterations for the optimization process.
- tol *float64*: tolerance for stopping criteria.
```
var err error
lambda := 0.1
alpha := 0.5
maxIter := 1000
tol := 1e-6

model := gorillaz.LinearRegression{}
err = model.FitElasticNet(X, Y, lambda, alpha, maxIter, tol)
```
#### FitPolynomial
Fit the model using polynomial regression by transforming features to polynomial space and fitting with linear regression.
- X *[][]float64*: independent variable data.
- Y *[][]float64*: dependent variable data.
- degree *int*: the degree of the polynomial features to generate.
```
var err error
degree := 3

model := gorillaz.LinearRegression{}
err = model.FitPolynomial(X, Y, degree)
```
#### FitBayesian
Fit the model using Bayesian linear regression with prior regularization.
- X *[][]float64*: independent variable data.
- Y *[][]float64*: dependent variable data.
- alpha *float64*: regularization strength for the prior precision term; must be positive.
```
var err error
alpha := 0.01

model := gorillaz.LinearRegression{}
err = model.FitBayesian(X, Y, alpha)
```
#### Predict
Make predictions using the trained linear regression model.
- X [][]float64: independent variable data for prediction.
- Returns *[][]float64*: predicted values for each target variable.
```
var err error
XNew := [][]float64{
    {1.0, 2.0},
    {3.0, 4.0},
    {5.0, 6.0},
}

model := gorillaz.LinearRegression{}
predictions, err := model.Predict(XNew)
```
#### RSquare
Calculate the coefficient of determination (R²) for the predictions made by the model.
- X *[][]float64*: independent variable data.
- Y *[][]float64*: true dependent variable data.
- Returns *float64*: R² score, representing the proportion of variance explained by the model.
```
var err error
XTest := [][]float64{
    {1.0, 2.0},
    {3.0, 4.0},
    {5.0, 6.0},
}
YTest := [][]float64{
    {1.1},
    {2.1},
    {3.1},
}

model := gorillaz.LinearRegression{}
score, err := model.RSquare(XTest, YTest)
```
## Classifiers
### Linear Classifiers
To perform any Linear Classification operation you need to use LinearClassifier struct. It stores the coefficients and the Intercept points.
```
model := gorillaz.LinearClassifier{}
```
#### FitLogisticClassifier
Fit the model using logistic regression for classification tasks.
- X *[][]float64*: independent variable data.
- Y *[][]float64*: target variable data in one-hot encoded format for multiclass classification.
- maxIter *int*: maximum number of iterations for gradient descent optimization.
- learningRate *float64*: step size for gradient descent.
- tol *float64*: tolerance for convergence criteria.
```
var err error
maxIter := 1000
learningRate := 0.01
tol := 1e-6

model := gorillaz.LinearClassifier{}
err = model.FitLogisticClassifier(X, Y, maxIter, learningRate, tol)
```
#### FitSVMClassifier
Fit the model using a Support Vector Machine (SVM) classifier with a linear kernel.
- X *[][]float64*: independent variable data.
- Y *[][]float64*: target variable data in one-hot encoded format for multiclass classification.
- learningRate *float64*: step size for gradient descent.
- regularizationParameter *float64*: penalty term controlling margin width and misclassification tolerance.
- maxIter *int*: maximum number of iterations for the training process.
```
var err error
learningRate := 0.01
regularizationParameter := 0.1
maxIter := 1000

model := gorillaz.LinearClassifier{}
err = model.FitSVMClassifier(X, Y, learningRate, regularizationParameter, maxIter)
```
#### Predict
Make predictions using the trained linear classifier.
- X *[][]float64*: independent variable data for prediction.
- Returns *[][]float64*: predicted probabilities or class scores for each class.
```
var err error
XNew := [][]float64{
    {1.0, 2.0},
    {3.0, 4.0},
    {5.0, 6.0},
}

model := gorillaz.LinearClassifier{}
predictions, err := model.Predict(XNew)
```
#### Accuracy
Calculate the accuracy of the model on the provided dataset.
- X *[][]float64*: independent variable data.
- Y *[][]float64*: true target variable data in one-hot encoded format.
- Returns *float64*: accuracy score, representing the proportion of correctly classified samples.
```
var err error

model := gorillaz.LinearClassifier{}
accuracy, err := model.Score(XTest, YTest)
```
### Tree Classifiers
To perform any Tree Classification operation you need to use DecisionTreeClassifier struct. It stores the nodes of it.
```
model := gorillaz.DecisionTreeClassifier{}
```
#### FitDecisionTreeClassifier
Fit the model using a decision tree for classification tasks.
- X *[][]float64*: independent variable data.
- Y *[][]float64*: target variable data.
- maxDepth *int*: maximum depth of the tree; controls overfitting.
- minSamples *int*: minimum number of samples required to split a node.
```
var err error
maxDepth := 5
minSamples := 2

model := gorillaz.DecisionTreeClassifier{}
err = model.FitDecisionTreeClassifier(X, Y, maxDepth, minSamples)
```
#### Predict
Make predictions using the trained decision tree classifier.
- X *[][]float64*: independent variable data for prediction.
- Returns *[]float64*: predicted class labels for each sample.
```
var err error

model := gorillaz.DecisionTreeClassifier{}
predictions, err := model.Predict(XNew)
```
### Naive Bayes Classifiers
To perform any Naive Bayes Classification operation you need to use MultinomialNaiveBayes struct.
```
model := gorillaz.MultinomialNaiveBayes{}
```
#### FitMultinomialNaiveBayes
Fit the model using a Multinomial Naive Bayes classifier for classification tasks.
- X *[][]float64*: independent variable data (typically counts or frequencies).
- Y *[]float64*: target variable data (class labels).
```
var err error

model := gorillaz.MultinomialNaiveBayes{}
err = model.FitMultinomialNaiveBayes(X, Y)
```
#### Predict
Make predictions using the trained Multinomial Naive Bayes classifier.
- X *[][]float64*: independent variable data for prediction.
- Returns *[]float64*: predicted class labels for each sample.
```
var err error

model := gorillaz.MultinomialNaiveBayes{}
predictions, err := model.Predict(XNew)
```
### Neighbors Classifiers
To perform any Neighbors Classification operation you need to use KNNClassifier struct.
```
model := gorillaz.KNNClassifier{}
```
#### FitKNNClassifier
Fit the model using the k-Nearest Neighbors (KNN) classifier by storing the training data.
- X *[][]float64*: independent variable data.
- Y *[][]float64*: target variable data (class labels).
- numberNeighbors *int*: number of nearest neighbors to consider for predictions.
```
var err error
numberNeighbors := 3

model := gorillaz.KNNClassifier{}
err = model.FitKNNClassifier(X, Y, numberNeighbors)
```
#### Predict
Make predictions using the trained k-Nearest Neighbors (KNN) classifier.
- X *[][]float64*: independent variable data for prediction.
- Returns *[]float64*: predicted class labels for each sample.
```
var err error

model := gorillaz.KNNClassifier{}
predictions, err := model.Predict(XNew)
```
## Utilities
### Model Valuation
#### Accuracy
Calculate the accuracy of predictions compared to true labels.
- yTrue *[]float64*: true class labels.
- yPred *[]float64*: predicted class labels.
- Returns *float64*: accuracy score, representing the proportion of correctly classified samples.
```
yTrue := []float64{1, 0, 1, 1, 0}
yPred := []float64{1, 0, 1, 0, 0}

accuracy := gorillaz.Accuracy(yTrue, yPred)
```
#### ConfusionMatrix
Generate a confusion matrix for evaluating classification performance.
- yTrue *[]float64*: true class labels.
- yPred *[]float64*: predicted class labels.
- Returns *map[string]int*: a confusion matrix with keys:
	"TP": True Positives
	"TN": True Negatives
	"FP": False Positives
	"FN": False Negatives
```
yTrue := []float64{1, 0, 1, 1, 0}
yPred := []float64{1, 0, 1, 0, 0}

confusionMatrix := gorillaz.ConfusionMatrix(yTrue, yPred)
```
#### Precision
Calculate the precision metric, which is the proportion of true positive predictions among all positive predictions.
- confusion *map[string]int*: a confusion matrix generated using ConfusionMatrix.
- Returns *float64*: precision score.
```
confusion := map[string]int{
    "TP": 10,
    "FP": 2,
    "TN": 5,
    "FN": 3,
}

precision := gorillaz.Precision(confusion)
```
#### Recall
Calculate the recall metric, which is the proportion of true positives identified among all actual positives.
- confusion *map[string]int*: a confusion matrix generated using ConfusionMatrix.
- Returns *float64*: recall score.
```
confusion := map[string]int{
    "TP": 10,
    "FP": 2,
    "TN": 5,
    "FN": 3,
}

recall := gorillaz.Recall(confusion)
```
#### F1Score
Calculate the F1 score, which is the harmonic mean of precision and recall.
- precision *float64*: precision score.
- recall *float64*: recall score.
- Returns *float64*: F1 score.
```
precision := 0.83
recall := 0.77

f1Score := gorillaz.F1Score(precision, recall)
```
### Save/Load
#### SaveToFile
Save a trained model to a file for later use.
- filePath *string*: path to the file where the model will be saved.
- model *interface{}*: the model object to be saved.
- Returns *error*: an error if the save operation fails.
```
var err error
filePath := "model.gob"

model := gorillaz.LinearRegression{}
// Assuming the model is trained...
err = gorillaz.SaveToFile(filePath, model)
if err != nil {
    fmt.Println("Error saving model:", err)
} else {
    fmt.Println("Model saved successfully!")
}
```
#### LoadFromFile
Load a previously saved model from a file.
- filePath *string*: path to the file where the model is stored.
- model *interface{}*: a reference to the model object where the data will be loaded.
- Returns *error*: an error if the load operation fails.
```
var err error
filePath := "model.gob"

var loadedModel gorillaz.LinearRegression
err = gorillaz.LoadFromFile(filePath, &loadedModel)
if err != nil {
    fmt.Println("Error loading model:", err)
} else {
    fmt.Println("Model loaded successfully!")
}
```
