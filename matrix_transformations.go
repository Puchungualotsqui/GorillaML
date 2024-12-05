package main

import (
	"runtime"
	"sync"
)

func addInterceptColumn(X [][]float64) [][]float64 {
	nRows := len(X)
	nCols := len(X[0]) + 1 // Add one column for the intercept

	// Initialize the result matrix
	XWithIntercept := make([][]float64, nRows)
	for i := range XWithIntercept {
		XWithIntercept[i] = make([]float64, nCols)
	}

	// Determine the number of goroutines based on available CPU cores
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
				XWithIntercept[i][0] = 1.0        // Intercept column is all ones
				copy(XWithIntercept[i][1:], X[i]) // Copy the original row data
			}
		}(start, end)
	}

	// Wait for all goroutines to finish
	wg.Wait()

	return XWithIntercept
}
