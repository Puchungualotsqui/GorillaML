package gorillaz

import (
	"encoding/gob"
	"fmt"
	"os"
)

// SaveToFile Generic function to save any model to a file
func SaveToFile(filePath string, model interface{}) error {
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("failed to create file: %v", err)
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	err = encoder.Encode(model)
	if err != nil {
		return fmt.Errorf("failed to encode model: %v", err)
	}

	return nil
}

// LoadFromFile Generic function to load any model from a file
func LoadFromFile(filePath string, model interface{}) error {
	file, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	err = decoder.Decode(model)
	if err != nil {
		return fmt.Errorf("failed to decode model: %v", err)
	}

	return nil
}

/* EXAMPLE
// SaveModel saves the trained model to a file
func (nb *MultinomialNaiveBayes) SaveModel(filePath string) error {
	return saveToFile(filePath, nb)
}

// LoadModel loads the trained model from a file
func (nb *MultinomialNaiveBayes) LoadModel(filePath string) error {
	return loadFromFile(filePath, nb)
}
*/
