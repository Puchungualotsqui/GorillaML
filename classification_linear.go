package gorillaz

import (
	"fmt"
	"math"
	"sync"
)

type LinearClassifier struct {
	Coefficients [][]float64 // Coefficients for each feature per class
	Intercept    []float64   // Intercept term for each class
}

func (lc *LinearClassifier) FitLogisticClassifier(X, Y [][]float64, maxIter int, learningRate float64, tol float64) error {
	if len(X) != len(Y) {
		return fmt.Errorf("number of rows in X and Y must match")
	}

	// Add intercept column to X
	XWithIntercept := addInterceptColumn(X)

	nSamples, nFeatures := len(XWithIntercept), len(XWithIntercept[0])
	nClasses := len(Y[0])

	// Initialize coefficients and intercepts for each class
	lc.Coefficients = make([][]float64, nClasses)
	lc.Intercept = make([]float64, nClasses)
	for i := range lc.Coefficients {
		lc.Coefficients[i] = make([]float64, nFeatures-1)
	}

	// Gradient descent
	for iter := 0; iter < maxIter; iter++ {
		maxChange := 0.0

		// Compute the gradient for each class
		for c := 0; c < nClasses; c++ {
			gradientW, gradientB := computeGradient(XWithIntercept, Y, lc.Coefficients[c], lc.Intercept[c], c)

			// Update coefficients and intercept using learning rate
			for j := 0; j < len(gradientW); j++ {
				change := learningRate * gradientW[j] / float64(nSamples)
				lc.Coefficients[c][j] += change
				maxChange = math.Max(maxChange, math.Abs(change))
			}
			interceptChange := learningRate * gradientB / float64(nSamples)
			lc.Intercept[c] += interceptChange
			maxChange = math.Max(maxChange, math.Abs(interceptChange))
		}

		// Check convergence
		if maxChange < tol {
			break
		}
	}

	return nil
}

func (lc *LinearClassifier) FitSVMClassifier(X, Y [][]float64, learningRate, regularizationParameter float64, maxIter int) error {
	if len(X) != len(Y) {
		return fmt.Errorf("number of rows in X and Y must match")
	}

	nSamples, nFeatures := len(X), len(X[0])
	numClasses := len(Y[0])

	// Initialize weights and biases for each class
	lc.Coefficients = make([][]float64, numClasses)
	lc.Intercept = make([]float64, numClasses)
	for i := 0; i < numClasses; i++ {
		lc.Coefficients[i] = make([]float64, nFeatures)
	}

	// WaitGroup to manage goroutines
	var wg sync.WaitGroup
	wg.Add(numClasses)

	for c := 0; c < numClasses; c++ {
		go func(c int) {
			defer wg.Done()
			// Gradient Descent for class c (One-vs-Rest)
			for iter := 0; iter < maxIter; iter++ {
				// Initialize gradient updates
				gradientW := make([]float64, nFeatures)
				gradientB := 0.0

				for i := 0; i < nSamples; i++ {
					// Create binary labels for class c (One-vs-Rest)
					yi := 1.0
					if Y[i][0] != float64(c) {
						yi = -1.0
					}

					// Compute linear function value
					linearOutput := lc.Intercept[c]
					for j := 0; j < nFeatures; j++ {
						linearOutput += lc.Coefficients[c][j] * X[i][j]
					}

					// Check if the sample is misclassified
					if yi*linearOutput < 1 {
						// Misclassified sample, update the gradients
						for j := 0; j < nFeatures; j++ {
							gradientW[j] += -yi * X[i][j]
						}
						gradientB += -yi
					}
				}

				// Update weights and bias with regularization term
				for j := 0; j < nFeatures; j++ {
					regularization := regularizationParameter * lc.Coefficients[c][j]
					lc.Coefficients[c][j] -= learningRate * (gradientW[j]/float64(nSamples) + regularization)
				}
				lc.Intercept[c] -= learningRate * (gradientB / float64(nSamples))
			}
		}(c)
	}

	wg.Wait()
	return nil
}

func (lc *LinearClassifier) Predict(X [][]float64) ([][]float64, error) {
	if lc.Coefficients == nil || lc.Intercept == nil {
		return nil, fmt.Errorf("model is not trained")
	}

	// Add intercept column to X
	XWithIntercept := addInterceptColumn(X)

	// Prepare predictions matrix
	nSamples := len(XWithIntercept)
	nClasses := len(lc.Coefficients)
	predictions := make([][]float64, nSamples)
	for i := range predictions {
		predictions[i] = make([]float64, nClasses)
	}

	// Compute predictions for each sample concurrently
	var wg sync.WaitGroup
	wg.Add(nSamples)
	for i := 0; i < nSamples; i++ {
		go func(i int) {
			defer wg.Done()
			computePredictions(XWithIntercept[i], lc.Coefficients, lc.Intercept, predictions[i])
		}(i)
	}
	wg.Wait()

	return predictions, nil
}

func (lc *LinearClassifier) Score(X, Y [][]float64) (float64, error) {
	// Predict the target values
	predictions, err := lc.Predict(X)
	if err != nil {
		return 0, err
	}

	nSamples := len(Y)
	nCorrect := 0

	// Determine the predicted class and compare to actual
	for i := 0; i < nSamples; i++ {
		trueClass := findTrueClass(Y[i])
		predictedClass := findPredictedClass(predictions[i])

		if predictedClass == trueClass {
			nCorrect++
		}
	}

	// Return accuracy
	return float64(nCorrect) / float64(nSamples), nil
}
