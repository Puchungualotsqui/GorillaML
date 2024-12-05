package gorillaz

import (
	"fmt"
	"math"
	"sync"
)

type MultinomialNaiveBayes struct {
	ClassLogPrior  []float64
	FeatureLogProb [][]float64
	Classes        []float64
}

// FitMultinomialNaiveBayes Fit trains the Multinomial Naive Bayes model on the given training data
func (nb *MultinomialNaiveBayes) FitMultinomialNaiveBayes(X [][]float64, Y []float64) error {
	if len(X) != len(Y) {
		return fmt.Errorf("number of rows in X and Y must match")
	}

	classCount := make(map[float64]int)
	featureCount := make(map[float64][]float64)
	nSamples := len(X)
	nFeatures := len(X[0])

	// Initialize class counts and feature counts
	for i := 0; i < nSamples; i++ {
		label := Y[i]
		classCount[label]++
		if _, exists := featureCount[label]; !exists {
			featureCount[label] = make([]float64, nFeatures)
		}
		for j := 0; j < nFeatures; j++ {
			featureCount[label][j] += X[i][j]
		}
	}

	// Calculate class log priors
	nClasses := len(classCount)
	nb.Classes = make([]float64, 0, nClasses)
	nb.ClassLogPrior = make([]float64, nClasses)
	for class, count := range classCount {
		nb.Classes = append(nb.Classes, class)
		nb.ClassLogPrior = append(nb.ClassLogPrior, math.Log(float64(count)/float64(nSamples)))
	}

	// Calculate feature log probabilities concurrently
	nb.FeatureLogProb = make([][]float64, nClasses)
	var wg sync.WaitGroup
	wg.Add(nClasses)
	for i, class := range nb.Classes {
		go func(i int, class float64) {
			defer wg.Done()
			classFeatureCount := featureCount[class]
			totalCount := 0.0
			for _, count := range classFeatureCount {
				totalCount += count
			}
			logProb := make([]float64, nFeatures)
			for j := 0; j < nFeatures; j++ {
				logProb[j] = math.Log((classFeatureCount[j] + 1) / (totalCount + float64(nFeatures)))
			}
			nb.FeatureLogProb[i] = logProb
		}(i, class)
	}
	wg.Wait()

	return nil
}

// Predict predicts the labels for the given test samples
func (nb *MultinomialNaiveBayes) Predict(X [][]float64) ([]float64, error) {
	if nb.ClassLogPrior == nil || nb.FeatureLogProb == nil {
		return nil, fmt.Errorf("model is not trained")
	}

	predictions := make([]float64, len(X))
	var wg sync.WaitGroup
	errChan := make(chan error, len(X))

	for i, sample := range X {
		wg.Add(1)
		go func(i int, sample []float64) {
			defer wg.Done()
			predictedLabel := nb.predictSingle(sample)
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
func (nb *MultinomialNaiveBayes) predictSingle(sample []float64) float64 {
	maxLogProb := math.Inf(-1)
	bestClass := nb.Classes[0]

	for i, class := range nb.Classes {
		logProb := nb.ClassLogPrior[i]
		for j := 0; j < len(sample); j++ {
			logProb += sample[j] * nb.FeatureLogProb[i][j]
		}
		if logProb > maxLogProb {
			maxLogProb = logProb
			bestClass = class
		}
	}

	return bestClass
}
