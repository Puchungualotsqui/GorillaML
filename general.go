package main

import (
	"fmt"
	"math"
)

// SoftThresholding applies the soft-thresholding operator
func SoftThresholding(z, lambda float64) float64 {
	if z > lambda {
		return z - lambda
	} else if z < -lambda {
		return z + lambda
	}
	return 0.0
}

func GeneratePolynomialFeatures(X [][]float64, degree int) ([][]float64, error) {
	if degree < 1 {
		return nil, fmt.Errorf("degree must be at least 1")
	}

	nSamples := len(X)
	nFeatures := len(X[0])
	expandedFeatures := [][]float64{}

	for i := 0; i < nSamples; i++ {
		row := []float64{}
		for j := 0; j < nFeatures; j++ {
			for d := 1; d <= degree; d++ {
				row = append(row, math.Pow(X[i][j], float64(d)))
			}
		}
		expandedFeatures = append(expandedFeatures, row)
	}

	return expandedFeatures, nil
}

func applyWeights(X, Y [][]float64, weights []float64, target int) ([][]float64, [][]float64) {
	nSamples, nFeatures := len(X), len(X[0])
	XWeighted := make([][]float64, nSamples)
	YWeighted := make([][]float64, nSamples)

	for i := 0; i < nSamples; i++ {
		XWeighted[i] = make([]float64, nFeatures)
		for j := 0; j < nFeatures; j++ {
			XWeighted[i][j] = X[i][j] * math.Sqrt(weights[i])
		}
		YWeighted[i] = []float64{Y[i][target] * math.Sqrt(weights[i])}
	}

	return XWeighted, YWeighted
}

func huberLossResidual(residual, delta float64) float64 {
	if math.Abs(residual) <= delta {
		return 1.0 // No penalty
	}
	return delta / math.Abs(residual)
}

func predictSingle(XRow []float64, intercept float64, coefficients []float64) float64 {
	prediction := intercept
	for j := 0; j < len(coefficients); j++ {
		prediction += XRow[j] * coefficients[j]
	}
	return prediction
}
