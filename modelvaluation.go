package gorillaz

import (
	"fmt"
)

// Accuracy calculates the accuracy of predictions compared to the true labels
func Accuracy(yTrue, yPred []float64) float64 {
	if len(yTrue) != len(yPred) {
		fmt.Println("Length of true labels and predicted labels must match.")
		return 0.0
	}

	correct := 0
	for i := range yTrue {
		if yTrue[i] == yPred[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(yTrue))
}

// ConfusionMatrix generates a confusion matrix for evaluating classification performance
func ConfusionMatrix(yTrue, yPred []float64) map[string]int {
	if len(yTrue) != len(yPred) {
		fmt.Println("Length of true labels and predicted labels must match.")
		return nil
	}

	confusion := map[string]int{
		"TP": 0, // True Positives
		"TN": 0, // True Negatives
		"FP": 0, // False Positives
		"FN": 0, // False Negatives
	}

	for i := range yTrue {
		if yTrue[i] == 1 && yPred[i] == 1 {
			confusion["TP"]++
		} else if yTrue[i] == 0 && yPred[i] == 0 {
			confusion["TN"]++
		} else if yTrue[i] == 0 && yPred[i] == 1 {
			confusion["FP"]++
		} else if yTrue[i] == 1 && yPred[i] == 0 {
			confusion["FN"]++
		}
	}

	return confusion
}

// Precision calculates the precision metric (TP / (TP + FP))
func Precision(confusion map[string]int) float64 {
	tp := float64(confusion["TP"])
	fp := float64(confusion["FP"])
	if tp+fp == 0 {
		return 0.0
	}
	return tp / (tp + fp)
}

// Recall calculates the recall metric (TP / (TP + FN))
func Recall(confusion map[string]int) float64 {
	tp := float64(confusion["TP"])
	fn := float64(confusion["FN"])
	if tp+fn == 0 {
		return 0.0
	}
	return tp / (tp + fn)
}

// F1Score calculates the F1 score (2 * (Precision * Recall) / (Precision + Recall))
func F1Score(precision, recall float64) float64 {
	if precision+recall == 0 {
		return 0.0
	}
	return 2 * (precision * recall) / (precision + recall)
}
