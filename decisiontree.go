package main

import (
	"fmt"
	"math"
	"sync"
)

type DecisionTreeNode struct {
	FeatureIndex int
	Threshold    float64
	Left         *DecisionTreeNode
	Right        *DecisionTreeNode
	Value        float64         // Only used for leaf nodes
	ClassCounts  map[float64]int // Class counts for leaf nodes (for multiclass)
	IsLeaf       bool
}

type DecisionTreeClassifier struct {
	Root       *DecisionTreeNode
	MaxDepth   int
	MinSamples int
}

func (dtc *DecisionTreeClassifier) FitDecisionTreeClassifier(X, Y [][]float64, maxDepth, minSamples int) error {
	if len(X) != len(Y) {
		return fmt.Errorf("number of rows in X and Y must match")
	}
	dtc.MaxDepth = maxDepth
	dtc.MinSamples = minSamples
	dtc.Root = dtc.buildTree(X, Y, 0)
	return nil
}

func (dtc *DecisionTreeClassifier) buildTree(X, Y [][]float64, depth int) *DecisionTreeNode {
	nSamples := len(Y)

	// Check stopping conditions
	if nSamples <= dtc.MinSamples || depth >= dtc.MaxDepth || dtc.isPure(Y) {
		leafValue := dtc.calculateLeafValue(Y)
		classCounts := dtc.calculateClassCounts(Y)
		return &DecisionTreeNode{
			Value:       leafValue,
			ClassCounts: classCounts,
			IsLeaf:      true,
		}
	}

	// Find the best split
	bestFeature, bestThreshold := dtc.findBestSplitParallel(X, Y)

	// Split dataset
	leftIndices, rightIndices := dtc.splitDataset(X, bestFeature, bestThreshold)
	leftX, leftY := dtc.extractSubset(X, Y, leftIndices)
	rightX, rightY := dtc.extractSubset(X, Y, rightIndices)

	// Create child nodes concurrently
	var leftChild, rightChild *DecisionTreeNode
	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		leftChild = dtc.buildTree(leftX, leftY, depth+1)
	}()

	go func() {
		defer wg.Done()
		rightChild = dtc.buildTree(rightX, rightY, depth+1)
	}()

	wg.Wait()

	return &DecisionTreeNode{
		FeatureIndex: bestFeature,
		Threshold:    bestThreshold,
		Left:         leftChild,
		Right:        rightChild,
	}
}

func (dtc *DecisionTreeClassifier) isPure(Y [][]float64) bool {
	firstClass := Y[0][0]
	for _, label := range Y {
		if label[0] != firstClass {
			return false
		}
	}
	return true
}

func (dtc *DecisionTreeClassifier) calculateLeafValue(Y [][]float64) float64 {
	classCounts := dtc.calculateClassCounts(Y)
	mostCommonClass := float64(-1)
	maxCount := 0
	for class, count := range classCounts {
		if count > maxCount {
			mostCommonClass = class
			maxCount = count
		}
	}
	return mostCommonClass
}

func (dtc *DecisionTreeClassifier) calculateClassCounts(Y [][]float64) map[float64]int {
	classCounts := make(map[float64]int)
	for _, label := range Y {
		classCounts[label[0]]++
	}
	return classCounts
}

func (dtc *DecisionTreeClassifier) findBestSplitParallel(X, Y [][]float64) (int, float64) {
	nFeatures := len(X[0])
	bestFeature := -1
	bestThreshold := 0.0
	bestImpurity := math.MaxFloat64

	var wg sync.WaitGroup
	mu := &sync.Mutex{}

	for featureIndex := 0; featureIndex < nFeatures; featureIndex++ {
		wg.Add(1)
		go func(featureIndex int) {
			defer wg.Done()
			thresholds := dtc.uniqueValues(X, featureIndex)
			for _, threshold := range thresholds {
				leftIndices, rightIndices := dtc.splitDataset(X, featureIndex, threshold)
				if len(leftIndices) == 0 || len(rightIndices) == 0 {
					continue
				}

				impurity := dtc.calculateImpurity(Y, leftIndices, rightIndices)

				mu.Lock()
				if impurity < bestImpurity {
					bestImpurity = impurity
					bestFeature = featureIndex
					bestThreshold = threshold
				}
				mu.Unlock()
			}
		}(featureIndex)
	}

	wg.Wait()

	return bestFeature, bestThreshold
}

func (dtc *DecisionTreeClassifier) calculateImpurity(Y [][]float64, leftIndices, rightIndices []int) float64 {
	totalSamples := len(leftIndices) + len(rightIndices)
	pLeft := float64(len(leftIndices)) / float64(totalSamples)
	pRight := float64(len(rightIndices)) / float64(totalSamples)

	impurityLeft := dtc.giniImpurity(Y, leftIndices)
	impurityRight := dtc.giniImpurity(Y, rightIndices)

	return pLeft*impurityLeft + pRight*impurityRight
}

func (dtc *DecisionTreeClassifier) giniImpurity(Y [][]float64, indices []int) float64 {
	classCounts := make(map[float64]int)
	for _, index := range indices {
		classCounts[Y[index][0]]++
	}
	total := float64(len(indices))
	gini := 1.0
	for _, count := range classCounts {
		p := float64(count) / total
		gini -= p * p
	}
	return gini
}

func (dtc *DecisionTreeClassifier) uniqueValues(X [][]float64, featureIndex int) []float64 {
	uniqueVals := make(map[float64]bool)
	for _, row := range X {
		uniqueVals[row[featureIndex]] = true
	}
	values := make([]float64, 0, len(uniqueVals))
	for val := range uniqueVals {
		values = append(values, val)
	}
	return values
}

func (dtc *DecisionTreeClassifier) splitDataset(X [][]float64, featureIndex int, threshold float64) ([]int, []int) {
	leftIndices, rightIndices := []int{}, []int{}
	for i, row := range X {
		if row[featureIndex] <= threshold {
			leftIndices = append(leftIndices, i)
		} else {
			rightIndices = append(rightIndices, i)
		}
	}
	return leftIndices, rightIndices
}

func (dtc *DecisionTreeClassifier) extractSubset(X, Y [][]float64, indices []int) ([][]float64, [][]float64) {
	subX := make([][]float64, len(indices))
	subY := make([][]float64, len(indices))
	for i, index := range indices {
		subX[i] = X[index]
		subY[i] = Y[index]
	}
	return subX, subY
}

func (dtc *DecisionTreeClassifier) Predict(X [][]float64) ([]float64, error) {
	if dtc.Root == nil {
		return nil, fmt.Errorf("model is not trained")
	}

	predictions := make([]float64, len(X))
	var wg sync.WaitGroup

	for i, sample := range X {
		wg.Add(1)
		go func(i int, sample []float64) {
			defer wg.Done()
			predictions[i] = dtc.predictSample(sample, dtc.Root)
		}(i, sample)
	}

	wg.Wait()
	return predictions, nil
}

func (dtc *DecisionTreeClassifier) predictSample(sample []float64, node *DecisionTreeNode) float64 {
	if node.IsLeaf {
		return node.Value
	}

	if sample[node.FeatureIndex] <= node.Threshold {
		return dtc.predictSample(sample, node.Left)
	}
	return dtc.predictSample(sample, node.Right)
}
