package gorillaz

import (
	"fmt"
	"math"
	"runtime"
	"sync"
)

func transposeMatrix(A [][]float64) [][]float64 {
	nRows := len(A)
	nCols := len(A[0])

	// Initialize transposed matrix
	result := make([][]float64, nCols)
	for i := range result {
		result[i] = make([]float64, nRows)
	}

	// Determine the number of goroutines based on available cores
	numGoroutines := runtime.NumCPU()
	chunkSize := (nCols + numGoroutines - 1) / numGoroutines // Divide columns into chunks

	// Use WaitGroup to synchronize goroutines
	var wg sync.WaitGroup

	for core := 0; core < numGoroutines; core++ {
		start := core * chunkSize
		end := start + chunkSize
		if end > nCols {
			end = nCols
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				for j := 0; j < nRows; j++ {
					result[i][j] = A[j][i]
				}
			}
		}(start, end)
	}

	// Wait for all goroutines to finish
	wg.Wait()

	return result
}

func matrixMultiplication(A, B [][]float64) [][]float64 {
	nRows := len(A)
	nCols := len(B[0])
	nInner := len(A[0])

	// Initialize result matrix
	result := make([][]float64, nRows)
	for i := range result {
		result[i] = make([]float64, nCols)
	}

	// Determine the number of goroutines based on available cores
	numCores := runtime.NumCPU()
	chunkSize := (nRows + numCores - 1) / numCores // Divide rows into chunks

	// Use WaitGroup to synchronize goroutines
	var wg sync.WaitGroup

	for core := 0; core < numCores; core++ {
		start := core * chunkSize
		end := start + chunkSize
		if end > nRows {
			end = nRows
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				for j := 0; j < nCols; j++ {
					for k := 0; k < nInner; k++ {
						result[i][j] += A[i][k] * B[k][j]
					}
				}
			}
		}(start, end)
	}

	// Wait for all goroutines to finish
	wg.Wait()

	return result
}

func multiplyMatrixVector(A [][]float64, b []float64) []float64 {
	nRows := len(A)
	nCols := len(A[0])

	// Initialize result vector
	result := make([]float64, nRows)

	// Determine the number of goroutines based on available cores
	numCores := runtime.NumCPU()
	chunkSize := (nRows + numCores - 1) / numCores // Divide rows into chunks

	// Use WaitGroup to synchronize goroutines
	var wg sync.WaitGroup

	for core := 0; core < numCores; core++ {
		start := core * chunkSize
		end := start + chunkSize
		if end > nRows {
			end = nRows
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				for j := 0; j < nCols; j++ {
					result[i] += A[i][j] * b[j]
				}
			}
		}(start, end)
	}

	// Wait for all goroutines to finish
	wg.Wait()

	return result
}

// invertMatrix computes the inverse of a square matrix using Gaussian Elimination with concurrency.
func invertMatrix(A [][]float64) ([][]float64, error) {
	n := len(A)
	if n == 0 || len(A[0]) != n {
		return nil, fmt.Errorf("matrix must be square")
	}

	// Create augmented matrix [A | I]
	augmented := make([][]float64, n)
	for i := 0; i < n; i++ {
		augmented[i] = make([]float64, 2*n)
		copy(augmented[i][:n], A[i]) // Copy A into the left half
		augmented[i][n+i] = 1        // Identity matrix on the right half
	}

	NumGoroutines := runtime.NumCPU() // Determine available CPU cores

	// Perform Gaussian Elimination
	for i := 0; i < n; i++ {
		// Make the diagonal element 1
		if augmented[i][i] == 0 {
			// Swap rows if the pivot is zero
			swapped := false
			for j := i + 1; j < n; j++ {
				if augmented[j][i] != 0 {
					augmented[i], augmented[j] = augmented[j], augmented[i]
					swapped = true
					break
				}
			}
			if !swapped {
				return nil, fmt.Errorf("matrix is singular and cannot be inverted")
			}
		}

		// Normalize the pivot row
		pivot := augmented[i][i]
		for j := 0; j < 2*n; j++ {
			augmented[i][j] /= pivot
		}

		// Make all other elements in the current column zero using concurrency
		var wg sync.WaitGroup
		chunkSize := (n + NumGoroutines - 1) / NumGoroutines // Divide rows among goroutines

		for core := 0; core < NumGoroutines; core++ {
			start := core * chunkSize
			end := start + chunkSize
			if end > n {
				end = n
			}

			wg.Add(1)
			go func(start, end, i int) {
				defer wg.Done()
				for k := start; k < end; k++ {
					if k != i {
						factor := augmented[k][i]
						for j := 0; j < 2*n; j++ {
							augmented[k][j] -= factor * augmented[i][j]
						}
					}
				}
			}(start, end, i)
		}

		wg.Wait() // Wait for all goroutines to finish
	}

	// Extract the right half of the augmented matrix as the inverse
	inverse := make([][]float64, n)
	for i := 0; i < n; i++ {
		inverse[i] = make([]float64, n)
		copy(inverse[i], augmented[i][n:])
	}

	return inverse, nil
}

func qRDecomposition(X [][]float64) (Q [][]float64, R [][]float64, err error) {
	nRows, nCols := len(X), len(X[0])
	Q = make([][]float64, nRows)
	R = make([][]float64, nCols)
	for i := range Q {
		Q[i] = make([]float64, nCols)
	}
	for i := range R {
		R[i] = make([]float64, nCols)
	}

	for j := 0; j < nCols; j++ {
		// Copy column j from X into q_j
		for i := 0; i < nRows; i++ {
			Q[i][j] = X[i][j]
		}

		// Subtract projections onto previous columns
		for k := 0; k < j; k++ {
			R[k][j] = 0
			for i := 0; i < nRows; i++ {
				R[k][j] += Q[i][k] * X[i][j]
			}
			for i := 0; i < nRows; i++ {
				Q[i][j] -= R[k][j] * Q[i][k]
			}
		}

		// Compute R[j][j] (norm of Q[:, j])
		norm := 0.0
		for i := 0; i < nRows; i++ {
			norm += Q[i][j] * Q[i][j]
		}
		R[j][j] = math.Sqrt(norm)

		if R[j][j] == 0 {
			return nil, nil, fmt.Errorf("matrix is singular")
		}

		// Normalize Q[:, j]
		for i := 0; i < nRows; i++ {
			Q[i][j] /= R[j][j]
		}
	}

	return Q, R, nil
}

func backSubstitution(R [][]float64, QtY []float64) []float64 {
	n := len(R)
	beta := make([]float64, n)

	for i := n - 1; i >= 0; i-- {
		sum := 0.0
		for j := i + 1; j < n; j++ {
			sum += R[i][j] * beta[j]
		}
		beta[i] = (QtY[i] - sum) / R[i][i]
	}

	return beta
}

// MinorMatrix generates a submatrix by removing the specified row and column.
func MinorMatrix(matrix [][]float64, excludeRow, excludeCol int) [][]float64 {
	n := len(matrix)
	minor := make([][]float64, n-1)
	rowIndex := 0

	for i := 0; i < n; i++ {
		if i == excludeRow {
			continue
		}
		minor[rowIndex] = make([]float64, n-1)
		colIndex := 0
		for j := 0; j < n; j++ {
			if j == excludeCol {
				continue
			}
			minor[rowIndex][colIndex] = matrix[i][j]
			colIndex++
		}
		rowIndex++
	}

	return minor
}

// determinant calculates the determinant of a square matrix using concurrency.
func determinant(matrix [][]float64) (float64, error) {
	n := len(matrix)
	if n == 0 || len(matrix[0]) != n {
		return 0, fmt.Errorf("matrix must be square")
	}

	// Base case for 1x1 matrix
	if n == 1 {
		return matrix[0][0], nil
	}

	// Base case for 2x2 matrix
	if n == 2 {
		return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0], nil
	}

	results := make(chan float64, n) // Channel to collect results
	var wg sync.WaitGroup

	// Perform Laplace expansion along the first row
	for col := 0; col < n; col++ {
		wg.Add(1)
		go func(col int) {
			defer wg.Done()

			// Compute the minor and determinant of the minor matrix
			minor := MinorMatrix(matrix, 0, col)
			subDeterminant, err := determinant(minor)
			if err != nil {
				panic(err) // Forward any unexpected errors
			}

			// Compute the cofactor and send result
			sign := 1.0
			if col%2 != 0 {
				sign = -1.0
			}
			results <- sign * matrix[0][col] * subDeterminant
		}(col)
	}

	// Wait for all goroutines to finish
	go func() {
		wg.Wait()
		close(results)
	}()

	// Sum up the results from all goroutines
	determinant := 0.0
	for result := range results {
		determinant += result
	}

	return determinant, nil
}
