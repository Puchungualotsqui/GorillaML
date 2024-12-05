package main

import (
	"math"
	"sync"
)

// computeGradient calculates the gradient for weights and intercept for a specific class.
func computeGradient(X [][]float64, Y [][]float64, coefficients []float64, intercept float64, classIndex int) ([]float64, float64) {
	nSamples, nFeatures := len(X), len(X[0])
	gradientW := make([]float64, nFeatures-1)
	gradientB := 0.0

	var wg sync.WaitGroup
	mu := &sync.Mutex{}

	wg.Add(nSamples)
	for i := 0; i < nSamples; i++ {
		go func(i int) {
			defer wg.Done()
			// Compute the predicted probability for the current class
			z := intercept
			for j := 1; j < nFeatures; j++ {
				z += coefficients[j-1] * X[i][j]
			}
			p := 1 / (1 + math.Exp(-z))

			// Compute the error term
			error := float64(Y[i][classIndex]) - p

			// Update gradients
			mu.Lock()
			gradientB += error
			for j := 1; j < nFeatures; j++ {
				gradientW[j-1] += error * X[i][j]
			}
			mu.Unlock()
		}(i)
	}
	wg.Wait()

	return gradientW, gradientB
}

// computePredictions calculates the predictions for each class for a given sample.
func computePredictions(sample []float64, coefficients [][]float64, intercepts []float64, predictions []float64) {
	sumExp := 0.0
	for c := 0; c < len(coefficients); c++ {
		// Compute the linear combination of weights and features
		z := intercepts[c]
		for j := 1; j < len(sample); j++ {
			z += coefficients[c][j-1] * sample[j]
		}
		predictions[c] = math.Exp(z)
		sumExp += predictions[c]
	}

	// Normalize to get class probabilities (softmax)
	for c := 0; c < len(coefficients); c++ {
		predictions[c] /= sumExp
	}
}

// findTrueClass finds the index of the true class in the target vector.
func findTrueClass(target []float64) int {
	for j := 0; j < len(target); j++ {
		if target[j] == 1 {
			return j
		}
	}
	return -1
}

// findPredictedClass finds the index of the predicted class based on the highest probability.
func findPredictedClass(predictions []float64) int {
	predictedClass := 0
	maxProb := predictions[0]
	for j := 1; j < len(predictions); j++ {
		if predictions[j] > maxProb {
			predictedClass = j
			maxProb = predictions[j]
		}
	}
	return predictedClass
}
