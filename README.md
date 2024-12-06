# GorillazML

Gorillaz is a machine learning library written in Go. It provides tools for building and evaluating various machine learning models, including regression, classification, and utilities for model persistence and evaluation metrics.

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

## Usage
### Regressions
####
### Classifiers
