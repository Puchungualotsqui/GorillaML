package main

import "fmt"

func main() {
	// Example dataset (binary classification)
	XTrain := [][]float64{
		{1.0, 2.0},
		{1.5, 1.8},
		{5.0, 8.0},
		{8.0, 8.0},
		{1.0, 0.6},
		{9.0, 11.0},
	}

	YTrain := [][]float64{
		{0},
		{0},
		{1},
		{1},
		{0},
		{1},
	}

	XTest := [][]float64{
		{1.2, 1.9},
		{6.0, 9.0},
	}

	// Create KNN classifier
	knn := KNNClassifier{}

	// Fit the model
	err := knn.FitKNNClassifier(XTrain, YTrain, 3)
	if err != nil {
		fmt.Printf("Error fitting model: %v\n", err)
		return
	}

	// Predict on the test data
	predictions, err := knn.Predict(XTest)
	if err != nil {
		fmt.Printf("Error predicting: %v\n", err)
		return
	}

	fmt.Println("Predictions:")
	for i, prediction := range predictions {
		fmt.Printf("Sample %d: %.0f\n", i, prediction)
	}
}
