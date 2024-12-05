package main

import (
	"fmt"
	"math"
)

type LinearRegression struct {
	Coefficients [][]float64 // Coefficients for each feature per target
	Intercept    []float64   // Intercept term for each target
	ExtraInfo    map[string]float64
}

func (lr *LinearRegression) FitOLS(X, Y [][]float64) error {
	var err error

	if len(X) != len(Y) {
		return fmt.Errorf("number of rows in xData and yData must match")
	}

	// Add intercept column to X
	XWithIntercept := addInterceptColumn(X)

	// Perform QR decomposition
	Q, R, err := qRDecomposition(XWithIntercept)
	if err != nil {
		return err
	}

	// Compute Q^T * Y for all target columns
	nTargets := len(Y[0])
	lr.Coefficients = make([][]float64, nTargets)
	lr.Intercept = make([]float64, nTargets)

	for j := 0; j < nTargets; j++ {
		// Extract target column
		targetColumn := make([]float64, len(Y))
		for i := 0; i < len(Y); i++ {
			targetColumn[i] = Y[i][j]
		}

		// Compute Q^T * y
		QtY := make([]float64, len(R))
		for i := 0; i < len(Q[0]); i++ {
			for k := 0; k < len(Q); k++ {
				QtY[i] += Q[k][i] * targetColumn[k]
			}
		}

		// Solve R * beta = Q^T * y
		beta := backSubstitution(R, QtY)

		// Store intercept and coefficients
		lr.Intercept[j] = beta[0]
		lr.Coefficients[j] = beta[1:]
	}

	return nil
}

func (lr *LinearRegression) FitLasso(X, Y [][]float64, lambda float64, maxIter int, tol float64) error {
	if len(X) != len(Y) {
		return fmt.Errorf("number of rows in X and Y must match")
	}

	// Add intercept column to X
	XWithIntercept := addInterceptColumn(X)

	nSamples, nFeatures := len(XWithIntercept), len(XWithIntercept[0])
	nTargets := len(Y[0])

	// Initialize coefficients
	lr.Coefficients = make([][]float64, nTargets)
	lr.Intercept = make([]float64, nTargets)

	for target := 0; target < nTargets; target++ {
		// Initialize coefficients for this target
		beta := make([]float64, nFeatures)

		// Perform Coordinate Descent
		for iter := 0; iter < maxIter; iter++ {
			maxChange := 0.0

			for j := 0; j < nFeatures; j++ {
				// Compute the partial residual excluding beta[j]
				residual := make([]float64, nSamples)
				for i := 0; i < nSamples; i++ {
					residual[i] = Y[i][target]
					for k := 0; k < nFeatures; k++ {
						if k != j {
							residual[i] -= XWithIntercept[i][k] * beta[k]
						}
					}
				}

				// Compute the numerator and denominator for beta[j]
				numerator, denominator := 0.0, 0.0
				for i := 0; i < nSamples; i++ {
					numerator += XWithIntercept[i][j] * residual[i]
					denominator += XWithIntercept[i][j] * XWithIntercept[i][j]
				}

				// Update beta[j] using soft-thresholding
				newBeta := softThresholding(numerator/denominator, lambda)
				maxChange = math.Max(maxChange, math.Abs(newBeta-beta[j]))
				beta[j] = newBeta
			}

			// Check convergence
			if maxChange < tol {
				break
			}
		}

		// Store intercept and coefficients
		lr.Intercept[target] = beta[0]
		lr.Coefficients[target] = beta[1:]
	}

	return nil
}

func (lr *LinearRegression) FitRidge(X, Y [][]float64, lambda float64) error {
	if len(X) != len(Y) {
		return fmt.Errorf("number of rows in xData and yData must match")
	}

	// Add intercept column to X
	XWithIntercept := addInterceptColumn(X)

	// Compute X^T * X
	Xt := transposeMatrix(XWithIntercept)
	XtX := matrixMultiplication(Xt, XWithIntercept)

	// Add lambda * I to X^T * X
	nFeatures := len(XtX)
	for i := 0; i < nFeatures; i++ {
		XtX[i][i] += lambda // Add regularization term
	}

	// Compute X^T * Y
	XtY := matrixMultiplication(Xt, Y)

	// Perform QR decomposition on (X^T * X + lambda * I)
	Q, R, err := qRDecomposition(XtX)
	if err != nil {
		return err
	}

	// Solve for coefficients using R and Q^T * (X^T * Y)
	nTargets := len(Y[0])
	lr.Coefficients = make([][]float64, nTargets)
	lr.Intercept = make([]float64, nTargets)

	for j := 0; j < nTargets; j++ {
		// Extract target column
		targetColumn := make([]float64, len(XtY))
		for i := 0; i < len(XtY); i++ {
			targetColumn[i] = XtY[i][j]
		}

		// Compute Q^T * XtY
		QtY := make([]float64, len(R))
		for i := 0; i < len(Q[0]); i++ {
			for k := 0; k < len(Q); k++ {
				QtY[i] += Q[k][i] * targetColumn[k]
			}
		}

		// Solve R * beta = Q^T * XtY
		beta := backSubstitution(R, QtY)

		// Store intercept and coefficients
		lr.Intercept[j] = beta[0]
		lr.Coefficients[j] = beta[1:]
	}

	return nil
}

func (lr *LinearRegression) FitElasticNet(X, Y [][]float64, lambda, alpha float64, maxIter int, tol float64) error {
	if len(X) != len(Y) {
		return fmt.Errorf("number of rows in X and Y must match")
	}

	// Add intercept column to X
	XWithIntercept := addInterceptColumn(X)

	nSamples, nFeatures := len(XWithIntercept), len(XWithIntercept[0])
	nTargets := len(Y[0])

	// Initialize coefficients
	lr.Coefficients = make([][]float64, nTargets)
	lr.Intercept = make([]float64, nTargets)

	// Elastic Net combines L1 and L2 penalties
	l1Penalty := lambda * alpha       // Lasso-like penalty
	l2Penalty := lambda * (1 - alpha) // Ridge-like penalty

	for target := 0; target < nTargets; target++ {
		// Initialize coefficients for this target
		beta := make([]float64, nFeatures)

		// Perform Coordinate Descent
		for iter := 0; iter < maxIter; iter++ {
			maxChange := 0.0

			for j := 0; j < nFeatures; j++ {
				// Compute the partial residual excluding beta[j]
				residual := make([]float64, nSamples)
				for i := 0; i < nSamples; i++ {
					residual[i] = Y[i][target]
					for k := 0; k < nFeatures; k++ {
						if k != j {
							residual[i] -= XWithIntercept[i][k] * beta[k]
						}
					}
				}

				// Compute the numerator and denominator for beta[j]
				numerator, denominator := 0.0, 0.0
				for i := 0; i < nSamples; i++ {
					numerator += XWithIntercept[i][j] * residual[i]
					denominator += XWithIntercept[i][j] * XWithIntercept[i][j]
				}

				// Add L2 regularization to the denominator
				denominator += 2 * l2Penalty

				// Update beta[j] using soft-thresholding for L1 regularization
				newBeta := softThresholding(numerator/denominator, l1Penalty/denominator)
				maxChange = math.Max(maxChange, math.Abs(newBeta-beta[j]))
				beta[j] = newBeta
			}

			// Check convergence
			if maxChange < tol {
				break
			}
		}

		// Store intercept and coefficients
		lr.Intercept[target] = beta[0]
		lr.Coefficients[target] = beta[1:]
	}

	return nil
}

func (lr *LinearRegression) FitRobust(X, Y [][]float64, delta float64, maxIter int, tol float64) error {
	if len(X) != len(Y) {
		return fmt.Errorf("number of rows in X and Y must match")
	}

	// Add intercept column to X
	XWithIntercept := addInterceptColumn(X)
	nSamples := len(XWithIntercept)
	nTargets := len(Y[0])

	// Initialize coefficients
	lr.Coefficients = make([][]float64, nTargets)
	lr.Intercept = make([]float64, nTargets)

	// Initialize weights
	weights := make([]float64, nSamples)
	for i := range weights {
		weights[i] = 1.0 // Initial weights
	}

	for target := 0; target < nTargets; target++ {
		for iter := 0; iter < maxIter; iter++ {
			// Apply weights
			XWeighted, YWeighted := applyWeights(XWithIntercept, Y, weights, target)

			// Solve weighted Ridge regression (regularized OLS)
			lambda := 1e-5 // Small regularization to prevent singularity
			err := lr.FitRidge(XWeighted, YWeighted, lambda)
			if err != nil {
				return fmt.Errorf("error in Ridge fitting: %v", err)
			}

			// Update weights based on residuals
			maxChange := 0.0
			for i := 0; i < nSamples; i++ {
				residual := Y[i][target] - predictSingle(XWithIntercept[i], lr.Intercept[target], lr.Coefficients[target])
				newWeight := huberLossResidual(residual, delta)
				maxChange = math.Max(maxChange, math.Abs(newWeight-weights[i]))
				weights[i] = newWeight
			}

			// Check for convergence
			if maxChange < tol {
				break
			}
		}
	}

	return nil
}

func (lr *LinearRegression) FitPolynomial(X, Y [][]float64, degree int) error {
	var err error
	var polyFeatures [][]float64
	// Generate polynomial features
	polyFeatures, err = generatePolynomialFeatures(X, degree)
	if err != nil {
		return err
	}

	// Fit the transformed features using Linear Regression
	err = lr.FitOLS(polyFeatures, Y)
	if err != nil {
		return err
	}

	return nil
}

func (lr *LinearRegression) FitBayesian(X, Y [][]float64, alpha float64) error {
	if len(X) != len(Y) {
		return fmt.Errorf("number of rows in X and Y must match")
	}

	if alpha <= 0 {
		return fmt.Errorf("regularization parameter Alpha must be positive")
	}

	// Add intercept column to X
	XWithIntercept := addInterceptColumn(X)

	// Compute X^T * X
	Xt := transposeMatrix(XWithIntercept)
	XtX := matrixMultiplication(Xt, XWithIntercept)

	// Add prior precision term (Alpha)
	for i := 0; i < len(XtX); i++ {
		XtX[i][i] += alpha
	}

	// Compute X^T * Y
	XtY := matrixMultiplication(Xt, Y)

	// Compute the posterior mean: (X^T X + Alpha I)⁻¹ X^T Y
	invXtX, err := invertMatrix(XtX)
	if err != nil {
		return fmt.Errorf("error inverting matrix: %v", err)
	}
	posteriorMean := matrixMultiplication(invXtX, XtY)

	// Store intercept and coefficients
	nTargets := len(Y[0])
	nFeatures := len(XWithIntercept[0])
	lr.Coefficients = make([][]float64, nTargets)
	lr.Intercept = make([]float64, nTargets)

	for j := 0; j < nTargets; j++ {
		lr.Intercept[j] = posteriorMean[0][j]
		lr.Coefficients[j] = make([]float64, nFeatures-1)
		for i := 1; i < nFeatures; i++ {
			lr.Coefficients[j][i-1] = posteriorMean[i][j]
		}
	}

	return nil
}

func (lr *LinearRegression) Predict(X [][]float64) ([][]float64, error) {
	if lr.Coefficients == nil || lr.Intercept == nil {
		return nil, fmt.Errorf("model is not trained")
	}

	// Add intercept column to X
	XWithIntercept := addInterceptColumn(X)

	// Prepare predictions matrix
	nRows := len(XWithIntercept)
	nTargets := len(lr.Coefficients)
	predictions := make([][]float64, nRows)
	for i := range predictions {
		predictions[i] = make([]float64, nTargets)
	}

	// Compute predictions for each target
	for i := 0; i < nRows; i++ {
		for j := 0; j < nTargets; j++ {
			// Compute dot product of row in XWithIntercept with coefficients (including intercept)
			for k, value := range XWithIntercept[i] {
				if k == 0 {
					predictions[i][j] += value * lr.Intercept[j] // First column corresponds to the intercept
				} else {
					predictions[i][j] += value * lr.Coefficients[j][k-1]
				}
			}
		}
	}

	return predictions, nil
}

func (lr *LinearRegression) Score(X, y [][]float64) (float64, error) {
	if lr.Coefficients == nil || lr.Intercept == nil {
		return 0, fmt.Errorf("model is not trained")
	}

	// Predict the target values
	predictions, err := lr.Predict(X)
	if err != nil {
		return 0, err
	}

	nRows := len(y)
	nTargets := len(y[0])

	// Compute total and residual sum of squares
	ssTotal := make([]float64, nTargets)
	ssResidual := make([]float64, nTargets)

	// Iterate over targets
	for j := 0; j < nTargets; j++ {
		// Calculate the mean of the true values for this target
		meanY := 0.0
		for i := 0; i < nRows; i++ {
			meanY += y[i][j]
		}
		meanY /= float64(nRows)

		// Compute SS_Total and SS_Residual
		for i := 0; i < nRows; i++ {
			ssTotal[j] += (y[i][j] - meanY) * (y[i][j] - meanY)
			ssResidual[j] += (y[i][j] - predictions[i][j]) * (y[i][j] - predictions[i][j])
		}
	}

	// Compute average R^2 across all targets
	rSquared := 0.0
	for j := 0; j < nTargets; j++ {
		rSquared += 1 - (ssResidual[j] / ssTotal[j])
	}
	rSquared /= float64(nTargets)

	return rSquared, nil
}
