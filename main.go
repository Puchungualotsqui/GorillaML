package main

import "fmt"

func main() {
	// Example dataset (multiclass classification)
	X := [][]float64{
		{1.0, 2.0},
		{1.5, 1.8},
		{5.0, 8.0},
		{8.0, 8.0},
		{1.0, 0.6},
		{9.0, 11.0},
	}

	Y := [][]float64{
		{1, 0, 0},
		{1, 0, 0},
		{0, 1, 0},
		{0, 1, 0},
		{0, 0, 1},
		{0, 0, 1},
	}

	// Create logistic regression model
	lr := LinearClassifier{}

	// Fit the model
	err := lr.FitLogisticClassifier(X, Y, 10000, 0.01, 1e-4)
	if err != nil {
		fmt.Printf("Error fitting model: %v\n", err)
		return
	}

	// Predict on the training data
	predictions, err := lr.Predict(X)
	if err != nil {
		fmt.Printf("Error predicting: %v\n", err)
		return
	}

	fmt.Println("Predictions:")
	for i, prediction := range predictions {
		fmt.Printf("Sample %d: %v\n", i, prediction)
	}

	// Score the model on training data
	accuracy, err := lr.Score(X, Y)
	if err != nil {
		fmt.Printf("Error scoring model: %v\n", err)
		return
	}

	fmt.Printf("Accuracy: %.2f%%\n", accuracy*100)
}
