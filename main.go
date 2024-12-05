package main

import "fmt"

func main() {
	// Example dataset (multiclass classification)
	XTrain := [][]float64{
		{2, 1, 0, 0},
		{1, 0, 3, 1},
		{0, 2, 1, 3},
		{0, 3, 0, 1},
		{2, 1, 0, 2},
	}

	YTrain := []float64{0, 1, 1, 0, 0}

	XTest := [][]float64{
		{1, 0, 1, 0},
		{0, 2, 0, 3},
	}

	// Create Multinomial Naive Bayes classifier
	nb := MultinomialNaiveBayes{}

	// Fit the model
	err := nb.Fit(XTrain, YTrain)
	if err != nil {
		fmt.Printf("Error fitting model: %v\n", err)
		return
	}

	// Predict on the test data
	predictions, err := nb.Predict(XTest)
	if err != nil {
		fmt.Printf("Error predicting: %v\n", err)
		return
	}

	fmt.Println("Predictions:")
	for i, prediction := range predictions {
		fmt.Printf("Sample %d: %.0f\n", i, prediction)
	}
}
