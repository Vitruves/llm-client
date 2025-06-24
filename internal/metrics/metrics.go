package metrics

import (
	"llm-client/internal/models"
	"math"
	"strings"
	"sync"
)

type Calculator struct {
	mu          sync.RWMutex
	config      *models.LiveMetrics
	classes     []string
	predictions []string
	actuals     []string
	confusionMx map[string]map[string]int
	total       int
}

func NewCalculator(config *models.LiveMetrics, classes []string) *Calculator {
	if config == nil || !config.Enabled {
		return nil
	}

	// Metrics require reference data - if GroundTruth is not specified, disable metrics
	if config.GroundTruth == "" {
		return nil
	}

	calc := &Calculator{
		config:      config,
		classes:     classes,
		confusionMx: make(map[string]map[string]int),
	}

	for _, class := range classes {
		calc.confusionMx[class] = make(map[string]int)
		for _, predClass := range classes {
			calc.confusionMx[class][predClass] = 0
		}
	}

	return calc
}

func (c *Calculator) AddResult(predicted, actual string) {
	if c == nil {
		return
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	c.predictions = append(c.predictions, predicted)
	c.actuals = append(c.actuals, actual)
	c.total++

	if c.confusionMx[actual] != nil {
		c.confusionMx[actual][predicted]++
	}
}

func (c *Calculator) GetCurrentMetric() float64 {
	if c == nil || c.total == 0 {
		return 0.0
	}

	c.mu.RLock()
	defer c.mu.RUnlock()

	switch c.config.Metric {
	case "accuracy":
		return c.calculateAccuracy()
	case "f1":
		return c.calculateF1()
	case "kappa":
		return c.calculateKappa()
	default:
		return c.calculateAccuracy()
	}
}

func (c *Calculator) GetMetricName() string {
	if c == nil {
		return "None"
	}

	switch c.config.Metric {
	case "accuracy":
		return "Accuracy"
	case "f1":
		avgType := c.config.Average
		if avgType == "" {
			avgType = "macro"
		}
		return "F1-" + avgType
	case "kappa":
		return "Kappa"
	default:
		return "Accuracy"
	}
}

func (c *Calculator) calculateAccuracy() float64 {
	if c.total == 0 {
		return 0.0
	}

	correct := 0
	for i := 0; i < len(c.predictions) && i < len(c.actuals); i++ {
		if c.predictions[i] == c.actuals[i] {
			correct++
		}
	}

	return float64(correct) / float64(c.total) * 100
}

func (c *Calculator) calculateF1() float64 {
	if c.total == 0 {
		return 0.0
	}

	switch c.config.Average {
	case "macro":
		return c.calculateMacroF1()
	case "micro":
		return c.calculateMicroF1()
	case "weighted":
		return c.calculateWeightedF1()
	default:
		return c.calculateMacroF1()
	}
}

func (c *Calculator) calculateMacroF1() float64 {
	perClassF1 := c.calculatePerClassF1()

	if len(perClassF1) == 0 {
		return 0.0
	}

	sum := 0.0
	count := 0
	for _, f1 := range perClassF1 {
		if !math.IsNaN(f1) {
			sum += f1
			count++
		}
	}

	if count == 0 {
		return 0.0
	}

	return sum / float64(count) * 100
}

func (c *Calculator) calculateMicroF1() float64 {
	totalTP := 0
	totalFP := 0
	totalFN := 0

	for _, class := range c.classes {
		tp, fp, fn := c.getClassMetrics(class)
		totalTP += tp
		totalFP += fp
		totalFN += fn
	}

	if totalTP == 0 {
		return 0.0
	}

	precision := float64(totalTP) / float64(totalTP+totalFP)
	recall := float64(totalTP) / float64(totalTP+totalFN)

	if precision+recall == 0 {
		return 0.0
	}

	return 2 * precision * recall / (precision + recall) * 100
}

func (c *Calculator) calculateWeightedF1() float64 {
	perClassF1 := c.calculatePerClassF1()

	weightedSum := 0.0
	totalSupport := 0

	for _, class := range c.classes {
		support := 0
		for actual, predictions := range c.confusionMx {
			if actual == class {
				for _, count := range predictions {
					support += count
				}
			}
		}

		f1 := perClassF1[class]
		if !math.IsNaN(f1) {
			weightedSum += f1 * float64(support)
		}
		totalSupport += support
	}

	if totalSupport == 0 {
		return 0.0
	}

	return weightedSum / float64(totalSupport) * 100
}

func (c *Calculator) calculatePerClassF1() map[string]float64 {
	result := make(map[string]float64)

	for _, class := range c.classes {
		tp, fp, fn := c.getClassMetrics(class)

		if tp == 0 {
			result[class] = 0.0
			continue
		}

		precision := float64(tp) / float64(tp+fp)
		recall := float64(tp) / float64(tp+fn)

		if precision+recall == 0 {
			result[class] = 0.0
		} else {
			result[class] = 2 * precision * recall / (precision + recall)
		}
	}

	return result
}

func (c *Calculator) getClassMetrics(class string) (tp, fp, fn int) {
	for actual, predictions := range c.confusionMx {
		for predicted, count := range predictions {
			if actual == class && predicted == class {
				tp += count
			} else if actual != class && predicted == class {
				fp += count
			} else if actual == class && predicted != class {
				fn += count
			}
		}
	}
	return tp, fp, fn
}

func (c *Calculator) calculateKappa() float64 {
	if c.total == 0 {
		return 0.0
	}

	po := c.calculateAccuracy() / 100

	pe := 0.0
	total := float64(c.total)

	for _, class := range c.classes {
		actualCount := 0
		predictedCount := 0

		for actual, predictions := range c.confusionMx {
			if actual == class {
				for _, count := range predictions {
					actualCount += count
				}
			}
			if predictions[class] > 0 {
				predictedCount += predictions[class]
			}
		}

		pActual := float64(actualCount) / total
		pPredicted := float64(predictedCount) / total
		pe += pActual * pPredicted
	}

	if pe == 1.0 {
		return 0.0
	}

	return (po - pe) / (1.0 - pe) * 100
}

func NormalizeLabel(label string) string {
	return strings.TrimSpace(label)
}
