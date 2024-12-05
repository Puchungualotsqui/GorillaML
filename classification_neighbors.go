package gorillaz

import (
	"fmt"
	"math"
	"sort"
	"sync"
)

type KNNClassifier struct {
	K      int         // Number of nearest neighbors
	XTrain [][]float64 // Training features
	YTrain []float64   // Training labels
}

type neighbor struct {
	index    int
	distance float64
}

// Fit stores the training data for future predictions
func (knn *KNNClassifier) FitKNNClassifier(X, Y [][]float64, numberNeighbors int) error {
	knn.K = numberNeighbors
	if len(X) != len(Y) {
		return fmt.Errorf("number of rows in X and Y must match")
	}
	knn.XTrain = X
	knn.YTrain = make([]float64, len(Y))
	for i := range Y {
		knn.YTrain[i] = Y[i][0]
	}
	return nil
}

// Predict predicts the labels for the given test samples
func (knn *KNNClassifier) Predict(X [][]float64) ([]float64, error) {
	if knn.XTrain == nil || knn.YTrain == nil {
		return nil, fmt.Errorf("model is not trained")
	}

	predictions := make([]float64, len(X))
	var wg sync.WaitGroup
	errChan := make(chan error, len(X))

	for i, sample := range X {
		wg.Add(1)
		go func(i int, sample []float64) {
			defer wg.Done()
			predictedLabel, err := knn.predictSingle(sample)
			if err != nil {
				errChan <- err
				return
			}
			predictions[i] = predictedLabel
		}(i, sample)
	}

	wg.Wait()
	close(errChan)

	if len(errChan) > 0 {
		return nil, <-errChan
	}

	return predictions, nil
}

// predictSingle predicts the label for a single test sample
func (knn *KNNClassifier) predictSingle(sample []float64) (float64, error) {
	distances := make([]neighbor, len(knn.XTrain))

	// Calculate the distance from the sample to each training instance
	for i, trainSample := range knn.XTrain {
		distance := euclideanDistance(sample, trainSample)
		distances[i] = neighbor{
			index:    i,
			distance: distance,
		}
	}

	// Sort by distance
	sort.Slice(distances, func(i, j int) bool {
		return distances[i].distance < distances[j].distance
	})

	// Get the labels of the K nearest neighbors
	labelCounts := make(map[float64]int)
	for i := 0; i < knn.K; i++ {
		label := knn.YTrain[distances[i].index]
		labelCounts[label]++
	}

	// Find the most common label
	maxCount := 0
	predictedLabel := 0.0
	for label, count := range labelCounts {
		if count > maxCount {
			maxCount = count
			predictedLabel = label
		}
	}

	return predictedLabel, nil
}

// euclideanDistance calculates the Euclidean distance between two vectors
func euclideanDistance(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		sum += (a[i] - b[i]) * (a[i] - b[i])
	}
	return math.Sqrt(sum)
}
