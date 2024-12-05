package main

import "fmt"

func main() {
	// Example dataset (multiclass classification)
	X := [][]float64{
		{2.0, 1.0},
		{10.0, 15.0},
		{6.0, 6.0},
		{-8.0, -8.0},
		{-1.0, -1.0},
		{-7.0, -7.0},
	}

	Y := [][]float64{
		{1},
		{1},
		{1},
		{1},
		{1},
		{1},
	}

	// Create Decision Tree classifier
	dtc := DecisionTreeClassifier{}

	// Fit the model
	err := dtc.FitDecisionTreeClassifier(X, Y, 3, 1)
	if err != nil {
		fmt.Printf("Error fitting model: %v\n", err)
		return
	}

	// Predict on the training data
	predictions, err := dtc.Predict(X)
	if err != nil {
		fmt.Printf("Error predicting: %v\n", err)
		return
	}

	fmt.Println("Predictions:")
	for i, prediction := range predictions {
		fmt.Printf("Sample %d: %.0f\n", i, prediction)
	}
}
