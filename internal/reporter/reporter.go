package reporter

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"time"

	"llm-client/internal/models"

	"github.com/parquet-go/parquet-go"
	"github.com/xuri/excelize/v2"
)

type Reporter struct {
	results []models.Result
	stats   *Stats
}

type Stats struct {
	Total       int
	Success     int
	Failed      int
	SuccessRate float64
	AvgTime     time.Duration
	MinTime     time.Duration
	MaxTime     time.Duration
	MedianTime  time.Duration
	P95Time     time.Duration
	P99Time     time.Duration
	StdDevTime  time.Duration
	TotalTime   time.Duration

	Accuracy          float64
	BalancedAccuracy  float64
	Precision         map[string]float64
	Recall            map[string]float64
	F1Score           map[string]float64
	Specificity       map[string]float64
	NPV               map[string]float64
	FPR               map[string]float64
	FNR               map[string]float64
	MacroF1           float64
	MicroF1           float64
	WeightedF1        float64
	MacroPrecision    float64
	MacroRecall       float64
	MicroPrecision    float64
	MicroRecall       float64
	WeightedPrecision float64
	WeightedRecall    float64
	Kappa             float64
	WeightedKappa     float64
	MCC               float64

	HammingLoss     float64
	JaccardScore    map[string]float64
	MacroJaccard    float64
	MicroJaccard    float64
	WeightedJaccard float64

	ConfusionMatrix      map[string]map[string]int
	ClassificationReport string

	ClassDistribution map[string]int
	ErrorDistribution map[string]int

	ThroughputPerSecond float64
	ThroughputPerMinute float64
	ThroughputPerHour   float64

	TimePercentiles map[string]time.Duration
	TimeBuckets     map[string]int

	ConsensusStats    *ConsensusStats
	HypothesisTests   *HypothesisTests
	DistributionTests *DistributionTests
}

type ConsensusStats struct {
	ItemsWithConsensus  int
	AvgConsensusRatio   float64
	MinConsensusRatio   float64
	MaxConsensusRatio   float64
	RepeatCount         int
	DistributionVariety map[int]int
}

type HypothesisTests struct {
	ChiSquareTest       *ChiSquareResult
	McNemerTest         *McNemerResult
	BinomialTest        *BinomialResult
	KolmogorovSmirnovTest *KSResult
}

type ChiSquareResult struct {
	Statistic float64 `json:"statistic"`
	PValue    float64 `json:"p_value"`
	DegreesOfFreedom int `json:"degrees_of_freedom"`
	CriticalValue float64 `json:"critical_value"`
	Significant bool `json:"significant"`
}

type McNemerResult struct {
	Statistic float64 `json:"statistic"`
	PValue    float64 `json:"p_value"`
	Significant bool `json:"significant"`
	B         int `json:"b"` // Disagreement count (predicted=0, actual=1)
	C         int `json:"c"` // Disagreement count (predicted=1, actual=0)
}

type BinomialResult struct {
	PValue      float64 `json:"p_value"`
	Successes   int     `json:"successes"`
	Trials      int     `json:"trials"`
	Probability float64 `json:"probability"`
	Significant bool    `json:"significant"`
}

type KSResult struct {
	Statistic   float64 `json:"statistic"`
	PValue      float64 `json:"p_value"`
	Significant bool    `json:"significant"`
}

type DistributionTests struct {
	ResponseTimeNormality *NormalityTest
	ClassBalance         *BalanceTest
	ErrorPattern         *PatternTest
}

type NormalityTest struct {
	ShapiroWilkW    float64 `json:"shapiro_wilk_w"`
	ShapiroWilkP    float64 `json:"shapiro_wilk_p"`
	JarqueBeraJB    float64 `json:"jarque_bera_jb"`
	JarqueBeraP     float64 `json:"jarque_bera_p"`
	IsNormal        bool    `json:"is_normal"`
}

type BalanceTest struct {
	ChiSquareStat   float64     `json:"chi_square_stat"`
	ChiSquareP      float64     `json:"chi_square_p"`
	ImbalanceRatio  float64     `json:"imbalance_ratio"`
	IsBalanced      bool        `json:"is_balanced"`
	ClassProportions map[string]float64 `json:"class_proportions"`
}

type PatternTest struct {
	EntropyScore     float64 `json:"entropy_score"`
	UniformityTest   float64 `json:"uniformity_test"`
	HasPattern       bool    `json:"has_pattern"`
	DominantPattern  string  `json:"dominant_pattern"`
}

func New(results []models.Result) *Reporter {
	return &Reporter{
		results: results,
		stats:   calculateStats(results),
	}
}

func (r *Reporter) GenerateText() string {
	var report strings.Builder

	r.writeHeader(&report)
	r.writeOverview(&report)
	r.writePerformanceMetrics(&report)
	r.writeTimeDistribution(&report)
	r.writeClassificationMetrics(&report)
	r.writeClassDistribution(&report)
	r.writeErrorAnalysis(&report)
	r.writeConsensusAnalysis(&report)
	r.writeSummary(&report)

	return report.String()
}

func (r *Reporter) writeHeader(report *strings.Builder) {
	report.WriteString("LLM CLASSIFICATION ANALYSIS REPORT\n")
	report.WriteString(strings.Repeat("-", 50) + "\n\n")
}

func (r *Reporter) writeOverview(report *strings.Builder) {
	report.WriteString("OVERVIEW\n")
	report.WriteString(fmt.Sprintf("  Total Items Processed: %d\n", r.stats.Total))
	report.WriteString(fmt.Sprintf("  Successful: %d\n", r.stats.Success))
	report.WriteString(fmt.Sprintf("  Failed: %d\n", r.stats.Failed))
	report.WriteString(fmt.Sprintf("  Success Rate: %.2f%%\n", r.stats.SuccessRate))
	report.WriteString(fmt.Sprintf("  Total Processing Time: %v\n", r.stats.TotalTime))
	report.WriteString("\n")
}

func (r *Reporter) writePerformanceMetrics(report *strings.Builder) {
	report.WriteString("PERFORMANCE METRICS\n")
	report.WriteString(fmt.Sprintf("  Throughput: %.2f items/sec, %.1f items/min, %.0f items/hour\n",
		r.stats.ThroughputPerSecond, r.stats.ThroughputPerMinute, r.stats.ThroughputPerHour))
	report.WriteString(fmt.Sprintf("  Average Response Time: %v\n", r.stats.AvgTime))
	report.WriteString(fmt.Sprintf("  Median Response Time: %v\n", r.stats.MedianTime))
	report.WriteString(fmt.Sprintf("  95th Percentile: %v\n", r.stats.P95Time))
	report.WriteString(fmt.Sprintf("  99th Percentile: %v\n", r.stats.P99Time))
	report.WriteString(fmt.Sprintf("  Min Response Time: %v\n", r.stats.MinTime))
	report.WriteString(fmt.Sprintf("  Max Response Time: %v\n", r.stats.MaxTime))
	report.WriteString(fmt.Sprintf("  Standard Deviation: %v\n", r.stats.StdDevTime))
	report.WriteString("\n")
}

func (r *Reporter) writeTimeDistribution(report *strings.Builder) {
	if len(r.stats.TimeBuckets) == 0 {
		return
	}

	report.WriteString("RESPONSE TIME DISTRIBUTION\n")
	for bucket, count := range r.stats.TimeBuckets {
		percentage := float64(count) / float64(r.stats.Total) * 100
		report.WriteString(fmt.Sprintf("  %s: %d items (%.1f%%)\n", bucket, count, percentage))
	}
	report.WriteString("\n")
}

func (r *Reporter) writeClassificationMetrics(report *strings.Builder) {
	if r.stats.Accuracy == 0 && len(r.stats.ClassDistribution) == 0 {
		return
	}

	report.WriteString("CLASSIFICATION METRICS\n")

	if r.stats.Accuracy > 0 {
		r.writeAccuracyMetrics(report)
		r.writeF1Analysis(report)
		r.writePrecisionRecallAnalysis(report)
		r.writeJaccardAnalysis(report)
	}

	if len(r.stats.Precision) > 0 {
		r.writePerClassAnalysis(report)
	}

	if len(r.stats.ConfusionMatrix) > 0 {
		r.writeConfusionMatrix(report)
	}

	if r.stats.ClassificationReport != "" {
		report.WriteString(r.stats.ClassificationReport)
		report.WriteString("\n")
	}
}

func (r *Reporter) writeAccuracyMetrics(report *strings.Builder) {
	report.WriteString(fmt.Sprintf("  Accuracy: %.2f%%\n", r.stats.Accuracy))
	report.WriteString(fmt.Sprintf("  Balanced Accuracy: %.2f%%\n", r.stats.BalancedAccuracy))
	report.WriteString(fmt.Sprintf("  Cohen's Kappa: %.4f\n", r.stats.Kappa))
	report.WriteString(fmt.Sprintf("  Weighted Kappa: %.4f\n", r.stats.WeightedKappa))
	report.WriteString(fmt.Sprintf("  Matthews Correlation Coefficient: %.4f\n", r.stats.MCC))
	report.WriteString(fmt.Sprintf("  Hamming Loss: %.4f\n", r.stats.HammingLoss))
	report.WriteString("\n")
}

func (r *Reporter) writeF1Analysis(report *strings.Builder) {
	report.WriteString("F1-SCORE ANALYSIS\n")
	report.WriteString(fmt.Sprintf("  Macro F1-Score: %.2f%%\n", r.stats.MacroF1))
	report.WriteString(fmt.Sprintf("  Micro F1-Score: %.2f%%\n", r.stats.MicroF1))
	report.WriteString(fmt.Sprintf("  Weighted F1-Score: %.2f%%\n", r.stats.WeightedF1))
	report.WriteString("\n")
}

func (r *Reporter) writePrecisionRecallAnalysis(report *strings.Builder) {
	report.WriteString("PRECISION & RECALL ANALYSIS\n")
	report.WriteString(fmt.Sprintf("  Macro Precision: %.2f%%\n", r.stats.MacroPrecision))
	report.WriteString(fmt.Sprintf("  Macro Recall: %.2f%%\n", r.stats.MacroRecall))
	report.WriteString(fmt.Sprintf("  Micro Precision: %.2f%%\n", r.stats.MicroPrecision))
	report.WriteString(fmt.Sprintf("  Micro Recall: %.2f%%\n", r.stats.MicroRecall))
	report.WriteString(fmt.Sprintf("  Weighted Precision: %.2f%%\n", r.stats.WeightedPrecision))
	report.WriteString(fmt.Sprintf("  Weighted Recall: %.2f%%\n", r.stats.WeightedRecall))
	report.WriteString("\n")
}

func (r *Reporter) writeJaccardAnalysis(report *strings.Builder) {
	report.WriteString("JACCARD INDEX ANALYSIS\n")
	report.WriteString(fmt.Sprintf("  Macro Jaccard: %.2f%%\n", r.stats.MacroJaccard))
	report.WriteString(fmt.Sprintf("  Micro Jaccard: %.2f%%\n", r.stats.MicroJaccard))
	report.WriteString(fmt.Sprintf("  Weighted Jaccard: %.2f%%\n", r.stats.WeightedJaccard))
	report.WriteString("\n")
}

func (r *Reporter) writePerClassAnalysis(report *strings.Builder) {
	report.WriteString("COMPREHENSIVE PER-CLASS ANALYSIS\n")

	for class := range r.stats.ClassDistribution {
		precision := r.stats.Precision[class]
		recall := r.stats.Recall[class]
		f1 := r.stats.F1Score[class]
		specificity := r.stats.Specificity[class]
		npv := r.stats.NPV[class]
		fpr := r.stats.FPR[class]
		fnr := r.stats.FNR[class]
		jaccard := r.stats.JaccardScore[class]
		support := r.stats.ClassDistribution[class]

		report.WriteString(fmt.Sprintf("  Class '%s' (Support: %d):\n", class, support))
		report.WriteString(fmt.Sprintf("    Precision: %.2f%%, Recall: %.2f%%, F1-Score: %.2f%%\n",
			precision*100, recall*100, f1*100))
		report.WriteString(fmt.Sprintf("    Specificity: %.2f%%, NPV: %.2f%%, Jaccard: %.2f%%\n",
			specificity*100, npv*100, jaccard*100))
		report.WriteString(fmt.Sprintf("    False Positive Rate: %.2f%%, False Negative Rate: %.2f%%\n",
			fpr*100, fnr*100))
	}
	report.WriteString("\n")
}

func (r *Reporter) writeConfusionMatrix(report *strings.Builder) {
	report.WriteString("CONFUSION MATRIX\n")

	var classes []string
	for class := range r.stats.ConfusionMatrix {
		classes = append(classes, class)
	}
	sort.Strings(classes)

	report.WriteString("Actual\\Predicted")
	for _, class := range classes {
		report.WriteString(fmt.Sprintf("%8s", class))
	}
	report.WriteString("\n")

	for _, actualClass := range classes {
		report.WriteString(fmt.Sprintf("%-15s", actualClass))
		for _, predictedClass := range classes {
			count := r.stats.ConfusionMatrix[actualClass][predictedClass]
			report.WriteString(fmt.Sprintf("%8d", count))
		}
		report.WriteString("\n")
	}
	report.WriteString("\n")
}

func (r *Reporter) writeClassDistribution(report *strings.Builder) {
	if len(r.stats.ClassDistribution) == 0 {
		return
	}

	report.WriteString("CLASS DISTRIBUTION\n")
	for class, count := range r.stats.ClassDistribution {
		percentage := float64(count) / float64(r.stats.Total) * 100
		report.WriteString(fmt.Sprintf("  Class '%s': %d items (%.1f%%)\n", class, count, percentage))
	}
	report.WriteString("\n")
}

func (r *Reporter) writeErrorAnalysis(report *strings.Builder) {
	if len(r.stats.ErrorDistribution) == 0 {
		return
	}

	report.WriteString("ERROR ANALYSIS\n")
	for errorType, count := range r.stats.ErrorDistribution {
		percentage := float64(count) / float64(r.stats.Failed) * 100
		report.WriteString(fmt.Sprintf("  %s: %d errors (%.1f%% of failures)\n", errorType, count, percentage))
	}
	report.WriteString("\n")
}

func (r *Reporter) writeConsensusAnalysis(report *strings.Builder) {
	if r.stats.ConsensusStats == nil {
		return
	}

	cs := r.stats.ConsensusStats
	report.WriteString("CONSENSUS ANALYSIS\n")
	report.WriteString(fmt.Sprintf("  Repeat Count: %d\n", cs.RepeatCount))
	report.WriteString(fmt.Sprintf("  Items with Consensus: %d\n", cs.ItemsWithConsensus))
	report.WriteString(fmt.Sprintf("  Average Consensus Ratio: %.2f%%\n", cs.AvgConsensusRatio*100))
	report.WriteString(fmt.Sprintf("  Min Consensus Ratio: %.2f%%\n", cs.MinConsensusRatio*100))
	report.WriteString(fmt.Sprintf("  Max Consensus Ratio: %.2f%%\n", cs.MaxConsensusRatio*100))

	if len(cs.DistributionVariety) > 0 {
		report.WriteString("  Distribution Variety:\n")
		for uniqueAnswers, count := range cs.DistributionVariety {
			report.WriteString(fmt.Sprintf("    %d unique answers: %d items\n", uniqueAnswers, count))
		}
	}
	report.WriteString("\n")
}

func (r *Reporter) writeSummary(report *strings.Builder) {
	report.WriteString("SUMMARY\n")

	switch {
	case r.stats.SuccessRate >= 95:
		report.WriteString("  Status: EXCELLENT - Very high success rate\n")
	case r.stats.SuccessRate >= 90:
		report.WriteString("  Status: GOOD - High success rate\n")
	case r.stats.SuccessRate >= 80:
		report.WriteString("  Status: ACCEPTABLE - Moderate success rate\n")
	default:
		report.WriteString("  Status: NEEDS IMPROVEMENT - Low success rate\n")
	}

	avgTimeMs := float64(r.stats.AvgTime.Nanoseconds()) / 1e6
	switch {
	case avgTimeMs < 1000:
		report.WriteString("  Performance: FAST - Sub-second average response\n")
	case avgTimeMs < 5000:
		report.WriteString("  Performance: GOOD - Fast response times\n")
	case avgTimeMs < 15000:
		report.WriteString("  Performance: MODERATE - Acceptable response times\n")
	default:
		report.WriteString("  Performance: SLOW - Consider optimization\n")
	}
}

func (r *Reporter) GenerateJSON() (string, error) {
	output := map[string]interface{}{
		"stats":   r.stats,
		"results": r.results,
	}

	data, err := json.MarshalIndent(output, "", "  ")
	if err != nil {
		return "", err
	}

	return string(data), nil
}

func (r *Reporter) SaveToFile(filename, format string) error {
	switch format {
	case "json":
		return r.saveJSON(filename)
	case "text":
		return r.saveText(filename)
	case "csv":
		return r.saveCSV(filename)
	case "xlsx":
		return r.saveExcel(filename)
	case "parquet":
		return r.saveParquet(filename)
	default:
		return fmt.Errorf("unsupported format: %s", format)
	}
}

func (r *Reporter) saveJSON(filename string) error {
	content, err := r.GenerateJSON()
	if err != nil {
		return err
	}
	return os.WriteFile(filename, []byte(content), 0644)
}

func (r *Reporter) saveText(filename string) error {
	content := r.GenerateText()
	return os.WriteFile(filename, []byte(content), 0644)
}

func (r *Reporter) saveCSV(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write summary statistics
	if err := r.writeCSVSummary(writer); err != nil {
		return err
	}

	// Write per-class metrics
	if err := r.writeCSVPerClassMetrics(writer); err != nil {
		return err
	}

	// Write hypothesis test results
	if err := r.writeCSVHypothesisTests(writer); err != nil {
		return err
	}

	return nil
}

func (r *Reporter) saveExcel(filename string) error {
	f := excelize.NewFile()
	defer f.Close()

	// Create sheets
	summarySheet := "Summary"
	perClassSheet := "PerClass"
	hypothesisSheet := "HypothesisTests"
	confusionSheet := "ConfusionMatrix"

	f.NewSheet(summarySheet)
	f.NewSheet(perClassSheet)
	f.NewSheet(hypothesisSheet)
	f.NewSheet(confusionSheet)

	// Write data to sheets
	if err := r.writeExcelSummary(f, summarySheet); err != nil {
		return err
	}
	if err := r.writeExcelPerClass(f, perClassSheet); err != nil {
		return err
	}
	if err := r.writeExcelHypothesis(f, hypothesisSheet); err != nil {
		return err
	}
	if err := r.writeExcelConfusion(f, confusionSheet); err != nil {
		return err
	}

	// Delete default sheet
	f.DeleteSheet("Sheet1")

	return f.SaveAs(filename)
}

func (r *Reporter) saveParquet(filename string) error {
	// Create a flattened structure for Parquet
	type ReportRow struct {
		Metric string  `parquet:"metric"`
		Value  float64 `parquet:"value"`
		Class  string  `parquet:"class,optional"`
		Type   string  `parquet:"type"`
	}

	var rows []ReportRow

	// Add summary metrics
	rows = append(rows, []ReportRow{
		{Metric: "accuracy", Value: r.stats.Accuracy, Type: "summary"},
		{Metric: "balanced_accuracy", Value: r.stats.BalancedAccuracy, Type: "summary"},
		{Metric: "macro_f1", Value: r.stats.MacroF1, Type: "summary"},
		{Metric: "micro_f1", Value: r.stats.MicroF1, Type: "summary"},
		{Metric: "weighted_f1", Value: r.stats.WeightedF1, Type: "summary"},
		{Metric: "kappa", Value: r.stats.Kappa, Type: "summary"},
		{Metric: "mcc", Value: r.stats.MCC, Type: "summary"},
		{Metric: "success_rate", Value: r.stats.SuccessRate, Type: "summary"},
		{Metric: "throughput_per_second", Value: r.stats.ThroughputPerSecond, Type: "performance"},
	}...)

	// Add per-class metrics
	for class, precision := range r.stats.Precision {
		rows = append(rows, []ReportRow{
			{Metric: "precision", Value: precision * 100, Class: class, Type: "per_class"},
			{Metric: "recall", Value: r.stats.Recall[class] * 100, Class: class, Type: "per_class"},
			{Metric: "f1_score", Value: r.stats.F1Score[class] * 100, Class: class, Type: "per_class"},
			{Metric: "specificity", Value: r.stats.Specificity[class] * 100, Class: class, Type: "per_class"},
		}...)
	}

	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := parquet.NewGenericWriter[ReportRow](file)
	defer writer.Close()

	_, err = writer.Write(rows)
	return err
}

func calculateStats(results []models.Result) *Stats {
	if len(results) == 0 {
		return createEmptyStats()
	}

	stats := &Stats{
		Total:             len(results),
		ClassDistribution: make(map[string]int),
		ErrorDistribution: make(map[string]int),
		Precision:         make(map[string]float64),
		Recall:            make(map[string]float64),
		F1Score:           make(map[string]float64),
		Specificity:       make(map[string]float64),
		NPV:               make(map[string]float64),
		FPR:               make(map[string]float64),
		FNR:               make(map[string]float64),
		JaccardScore:      make(map[string]float64),
		ConfusionMatrix:   make(map[string]map[string]int),
		TimePercentiles:   make(map[string]time.Duration),
		TimeBuckets:       make(map[string]int),
	}

	predictions, actuals, times, hasGroundTruth := processResults(results, stats)

	stats.Success = len(predictions)
	stats.Failed = len(results) - stats.Success
	stats.SuccessRate = float64(stats.Success) / float64(len(results)) * 100

	calculateTimeStatistics(stats, times)

	if hasGroundTruth && len(predictions) == len(actuals) {
		calculateClassificationMetrics(stats, predictions, actuals)
	}

	stats.ConsensusStats = calculateConsensusStats(results)

	return stats
}

func createEmptyStats() *Stats {
	return &Stats{
		ClassDistribution: make(map[string]int),
		ErrorDistribution: make(map[string]int),
		Precision:         make(map[string]float64),
		Recall:            make(map[string]float64),
		F1Score:           make(map[string]float64),
		Specificity:       make(map[string]float64),
		NPV:               make(map[string]float64),
		FPR:               make(map[string]float64),
		FNR:               make(map[string]float64),
		JaccardScore:      make(map[string]float64),
		ConfusionMatrix:   make(map[string]map[string]int),
		TimePercentiles:   make(map[string]time.Duration),
		TimeBuckets:       make(map[string]int),
	}
}

func processResults(results []models.Result, stats *Stats) ([]string, []string, []time.Duration, bool) {
	var predictions, actuals []string
	var times []time.Duration
	var hasGroundTruth bool

	for _, result := range results {
		times = append(times, result.ResponseTime)
		stats.TotalTime += result.ResponseTime

		if result.Success {
			stats.ClassDistribution[result.FinalAnswer]++
			predictions = append(predictions, result.FinalAnswer)

			if result.GroundTruth != "" {
				actuals = append(actuals, result.GroundTruth)
				hasGroundTruth = true
			}
		} else {
			errorType := categorizeError(result.Error)
			stats.ErrorDistribution[errorType]++
		}
	}

	return predictions, actuals, times, hasGroundTruth
}

func categorizeError(errorMsg string) string {
	errorMsg = strings.ToLower(errorMsg)

	switch {
	case strings.Contains(errorMsg, "timeout"):
		return "timeout"
	case strings.Contains(errorMsg, "connection"):
		return "connection"
	case strings.Contains(errorMsg, "parse"):
		return "parsing"
	default:
		return "other"
	}
}

func calculateTimeStatistics(stats *Stats, times []time.Duration) {
	if len(times) == 0 {
		return
	}

	sort.Slice(times, func(i, j int) bool {
		return times[i] < times[j]
	})

	stats.AvgTime = stats.TotalTime / time.Duration(len(times))
	stats.MinTime = times[0]
	stats.MaxTime = times[len(times)-1]
	stats.MedianTime = times[len(times)/2]
	stats.P95Time = times[int(float64(len(times))*0.95)]
	stats.P99Time = times[int(float64(len(times))*0.99)]

	calculateStandardDeviation(stats, times)
	calculatePercentiles(stats, times)
	calculateTimeBuckets(stats, times)
	calculateThroughput(stats)
}

func calculateStandardDeviation(stats *Stats, times []time.Duration) {
	var sumSquaredDiff float64
	avgNanos := float64(stats.AvgTime.Nanoseconds())

	for _, t := range times {
		diff := float64(t.Nanoseconds()) - avgNanos
		sumSquaredDiff += diff * diff
	}

	variance := sumSquaredDiff / float64(len(times))
	stats.StdDevTime = time.Duration(math.Sqrt(variance))
}

func calculatePercentiles(stats *Stats, times []time.Duration) {
	stats.TimePercentiles["P50"] = times[len(times)/2]
	stats.TimePercentiles["P75"] = times[int(float64(len(times))*0.75)]
	stats.TimePercentiles["P90"] = times[int(float64(len(times))*0.90)]
	stats.TimePercentiles["P95"] = stats.P95Time
	stats.TimePercentiles["P99"] = stats.P99Time
}

func calculateTimeBuckets(stats *Stats, times []time.Duration) {
	for _, t := range times {
		ms := float64(t.Nanoseconds()) / 1e6

		switch {
		case ms < 100:
			stats.TimeBuckets["< 100ms"]++
		case ms < 500:
			stats.TimeBuckets["100-500ms"]++
		case ms < 1000:
			stats.TimeBuckets["500ms-1s"]++
		case ms < 5000:
			stats.TimeBuckets["1-5s"]++
		case ms < 15000:
			stats.TimeBuckets["5-15s"]++
		default:
			stats.TimeBuckets["> 15s"]++
		}
	}
}

func calculateThroughput(stats *Stats) {
	if stats.TotalTime.Seconds() > 0 {
		stats.ThroughputPerSecond = float64(stats.Total) / stats.TotalTime.Seconds()
		stats.ThroughputPerMinute = stats.ThroughputPerSecond * 60
		stats.ThroughputPerHour = stats.ThroughputPerSecond * 3600
	}
}

func calculateClassificationMetrics(stats *Stats, predictions, actuals []string) {
	stats.Accuracy = calculateAccuracy(predictions, actuals)
	stats.BalancedAccuracy = calculateBalancedAccuracy(predictions, actuals)
	stats.Kappa = calculateKappa(predictions, actuals)
	stats.WeightedKappa = calculateKappa(predictions, actuals)
	stats.MCC = calculateMCC(predictions, actuals)
	stats.HammingLoss = calculateHammingLoss(predictions, actuals)

	stats.Precision, stats.Recall, stats.F1Score = calculatePerClassMetrics(predictions, actuals)
	stats.Specificity, stats.NPV, stats.FPR, stats.FNR = calculateAdvancedPerClassMetrics(predictions, actuals)
	stats.JaccardScore = calculateJaccardPerClass(predictions, actuals)
	stats.ConfusionMatrix = calculateConfusionMatrix(predictions, actuals)

	calculateAverageMetrics(stats, predictions, actuals)
	stats.ClassificationReport = generateClassificationReport(stats, predictions, actuals)
}

func calculateAverageMetrics(stats *Stats, predictions, actuals []string) {
	stats.MacroF1 = calculateMacroAverage(stats.F1Score)
	stats.MicroF1 = calculateMicroF1(predictions, actuals)
	stats.WeightedF1 = calculateWeightedAverage(stats.F1Score, predictions, actuals)
	stats.MacroPrecision = calculateMacroAverage(stats.Precision)
	stats.MacroRecall = calculateMacroAverage(stats.Recall)
	stats.MicroPrecision, stats.MicroRecall = calculateMicroPrecisionRecall(predictions, actuals)
	stats.WeightedPrecision = calculateWeightedAverage(stats.Precision, predictions, actuals)
	stats.WeightedRecall = calculateWeightedAverage(stats.Recall, predictions, actuals)
	stats.MacroJaccard = calculateMacroAverage(stats.JaccardScore)
	stats.MicroJaccard = calculateMicroJaccard(predictions, actuals)
	stats.WeightedJaccard = calculateWeightedAverage(stats.JaccardScore, predictions, actuals)
}

func calculateAccuracy(predictions, actuals []string) float64 {
	if len(predictions) != len(actuals) || len(predictions) == 0 {
		return 0.0
	}

	correct := 0
	for i := 0; i < len(predictions); i++ {
		if predictions[i] == actuals[i] {
			correct++
		}
	}

	return float64(correct) / float64(len(predictions)) * 100
}

func calculateBalancedAccuracy(predictions, actuals []string) float64 {
	classSet := getUniqueClasses(actuals)
	var recallSum float64
	classCount := 0

	for class := range classSet {
		tp, _, fn := getClassMetrics(class, predictions, actuals)
		if tp+fn > 0 {
			recall := float64(tp) / float64(tp+fn)
			recallSum += recall
			classCount++
		}
	}

	if classCount == 0 {
		return 0.0
	}

	return (recallSum / float64(classCount)) * 100
}

func calculateKappa(predictions, actuals []string) float64 {
	if len(predictions) != len(actuals) || len(predictions) == 0 {
		return 0.0
	}

	classSet := getUniqueClasses(append(predictions, actuals...))
	classes := make([]string, 0, len(classSet))
	for class := range classSet {
		classes = append(classes, class)
	}

	po := calculateAccuracy(predictions, actuals) / 100

	pe := 0.0
	total := float64(len(predictions))

	for _, class := range classes {
		actualCount := countClassOccurrences(class, actuals)
		predictedCount := countClassOccurrences(class, predictions)

		pActual := float64(actualCount) / total
		pPredicted := float64(predictedCount) / total
		pe += pActual * pPredicted
	}

	if pe == 1.0 {
		return 0.0
	}

	return (po - pe) / (1.0 - pe)
}

func calculateMCC(predictions, actuals []string) float64 {
	if len(predictions) != len(actuals) || len(predictions) == 0 {
		return 0.0
	}

	classSet := getUniqueClasses(actuals)
	var mccSum float64
	classCount := 0

	for class := range classSet {
		tp, fp, fn := getClassMetrics(class, predictions, actuals)
		tn := len(predictions) - tp - fp - fn

		denominator := math.Sqrt(float64((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
		if denominator == 0 {
			continue
		}

		mcc := float64(tp*tn-fp*fn) / denominator
		mccSum += mcc
		classCount++
	}

	if classCount == 0 {
		return 0.0
	}

	return mccSum / float64(classCount)
}

func calculateHammingLoss(predictions, actuals []string) float64 {
	if len(predictions) != len(actuals) || len(predictions) == 0 {
		return 0.0
	}

	incorrect := 0
	for i := 0; i < len(predictions); i++ {
		if predictions[i] != actuals[i] {
			incorrect++
		}
	}

	return float64(incorrect) / float64(len(predictions))
}

func calculatePerClassMetrics(predictions, actuals []string) (map[string]float64, map[string]float64, map[string]float64) {
	precision := make(map[string]float64)
	recall := make(map[string]float64)
	f1Score := make(map[string]float64)

	classSet := getUniqueClasses(append(predictions, actuals...))

	for class := range classSet {
		tp, fp, fn := getClassMetrics(class, predictions, actuals)

		if tp+fp > 0 {
			precision[class] = float64(tp) / float64(tp+fp)
		}

		if tp+fn > 0 {
			recall[class] = float64(tp) / float64(tp+fn)
		}

		if precision[class]+recall[class] > 0 {
			f1Score[class] = 2 * precision[class] * recall[class] / (precision[class] + recall[class])
		}
	}

	return precision, recall, f1Score
}

func calculateAdvancedPerClassMetrics(predictions, actuals []string) (map[string]float64, map[string]float64, map[string]float64, map[string]float64) {
	specificity := make(map[string]float64)
	npv := make(map[string]float64)
	fpr := make(map[string]float64)
	fnr := make(map[string]float64)

	classSet := getUniqueClasses(actuals)

	for class := range classSet {
		tp, fp, fn := getClassMetrics(class, predictions, actuals)
		tn := len(predictions) - tp - fp - fn

		if tn+fp > 0 {
			specificity[class] = float64(tn) / float64(tn+fp)
		}

		if tn+fn > 0 {
			npv[class] = float64(tn) / float64(tn+fn)
		}

		if fp+tn > 0 {
			fpr[class] = float64(fp) / float64(fp+tn)
		}

		if fn+tp > 0 {
			fnr[class] = float64(fn) / float64(fn+tp)
		}
	}

	return specificity, npv, fpr, fnr
}

func calculateJaccardPerClass(predictions, actuals []string) map[string]float64 {
	jaccard := make(map[string]float64)
	classSet := getUniqueClasses(actuals)

	for class := range classSet {
		tp, fp, fn := getClassMetrics(class, predictions, actuals)
		if tp+fp+fn > 0 {
			jaccard[class] = float64(tp) / float64(tp+fp+fn)
		}
	}

	return jaccard
}

func calculateConfusionMatrix(predictions, actuals []string) map[string]map[string]int {
	matrix := make(map[string]map[string]int)
	classSet := getUniqueClasses(append(predictions, actuals...))

	for actual := range classSet {
		matrix[actual] = make(map[string]int)
		for predicted := range classSet {
			matrix[actual][predicted] = 0
		}
	}

	for i := 0; i < len(predictions); i++ {
		actual := actuals[i]
		predicted := predictions[i]
		matrix[actual][predicted]++
	}

	return matrix
}

func calculateMacroAverage(values map[string]float64) float64 {
	if len(values) == 0 {
		return 0.0
	}

	sum := 0.0
	count := 0
	for _, value := range values {
		if !math.IsNaN(value) {
			sum += value
			count++
		}
	}

	if count == 0 {
		return 0.0
	}

	return (sum / float64(count)) * 100
}

func calculateMicroF1(predictions, actuals []string) float64 {
	classSet := getUniqueClasses(append(predictions, actuals...))

	totalTP := 0
	totalFP := 0
	totalFN := 0

	for class := range classSet {
		tp, fp, fn := getClassMetrics(class, predictions, actuals)
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

func calculateMicroPrecisionRecall(predictions, actuals []string) (float64, float64) {
	classSet := getUniqueClasses(actuals)

	totalTP := 0
	totalFP := 0
	totalFN := 0

	for class := range classSet {
		tp, fp, fn := getClassMetrics(class, predictions, actuals)
		totalTP += tp
		totalFP += fp
		totalFN += fn
	}

	var precision, recall float64

	if totalTP+totalFP > 0 {
		precision = float64(totalTP) / float64(totalTP+totalFP) * 100
	}

	if totalTP+totalFN > 0 {
		recall = float64(totalTP) / float64(totalTP+totalFN) * 100
	}

	return precision, recall
}

func calculateWeightedAverage(values map[string]float64, predictions, actuals []string) float64 {
	if len(values) == 0 {
		return 0.0
	}

	support := make(map[string]int)
	for _, actual := range actuals {
		support[actual]++
	}

	weightedSum := 0.0
	totalSupport := 0

	for class, value := range values {
		classSupport := support[class]
		if !math.IsNaN(value) {
			weightedSum += value * float64(classSupport)
		}
		totalSupport += classSupport
	}

	if totalSupport == 0 {
		return 0.0
	}

	return (weightedSum / float64(totalSupport)) * 100
}

func calculateMicroJaccard(predictions, actuals []string) float64 {
	classSet := getUniqueClasses(actuals)

	totalTP := 0
	totalFP := 0
	totalFN := 0

	for class := range classSet {
		tp, fp, fn := getClassMetrics(class, predictions, actuals)
		totalTP += tp
		totalFP += fp
		totalFN += fn
	}

	if totalTP+totalFP+totalFN > 0 {
		return float64(totalTP) / float64(totalTP+totalFP+totalFN) * 100
	}

	return 0.0
}

func generateClassificationReport(stats *Stats, predictions, actuals []string) string {
	var report strings.Builder

	report.WriteString("DETAILED CLASSIFICATION REPORT\n")
	report.WriteString(strings.Repeat("-", 80) + "\n")
	report.WriteString(fmt.Sprintf("%-10s %10s %10s %10s %10s %10s %10s %10s\n",
		"Class", "Precision", "Recall", "F1-Score", "Specificity", "NPV", "Jaccard", "Support"))
	report.WriteString(strings.Repeat("-", 80) + "\n")

	support := make(map[string]int)
	for _, actual := range actuals {
		support[actual]++
	}

	for class := range stats.ClassDistribution {
		precision := stats.Precision[class] * 100
		recall := stats.Recall[class] * 100
		f1 := stats.F1Score[class] * 100
		specificity := stats.Specificity[class] * 100
		npv := stats.NPV[class] * 100
		jaccard := stats.JaccardScore[class] * 100
		classSupport := support[class]

		report.WriteString(fmt.Sprintf("%-10s %9.2f%% %9.2f%% %9.2f%% %10.2f%% %9.2f%% %9.2f%% %10d\n",
			class, precision, recall, f1, specificity, npv, jaccard, classSupport))
	}

	report.WriteString(strings.Repeat("-", 80) + "\n")
	report.WriteString(fmt.Sprintf("%-10s %9.2f%% %9.2f%% %9.2f%% %10s %9s %9.2f%% %10d\n",
		"macro avg", stats.MacroPrecision, stats.MacroRecall, stats.MacroF1,
		"-", "-", stats.MacroJaccard, len(actuals)))
	report.WriteString(fmt.Sprintf("%-10s %9.2f%% %9.2f%% %9.2f%% %10s %9s %9.2f%% %10d\n",
		"micro avg", stats.MicroPrecision, stats.MicroRecall, stats.MicroF1,
		"-", "-", stats.MicroJaccard, len(actuals)))
	report.WriteString(fmt.Sprintf("%-10s %9.2f%% %9.2f%% %9.2f%% %10s %9s %9.2f%% %10d\n",
		"weighted", stats.WeightedPrecision, stats.WeightedRecall, stats.WeightedF1,
		"-", "-", stats.WeightedJaccard, len(actuals)))

	return report.String()
}

func calculateConsensusStats(results []models.Result) *ConsensusStats {
	var consensusResults []models.Result
	var repeatCount int

	for _, result := range results {
		if result.Consensus != nil {
			consensusResults = append(consensusResults, result)
			if repeatCount == 0 {
				repeatCount = result.Consensus.Total
			}
		}
	}

	if len(consensusResults) == 0 {
		return nil
	}

	stats := &ConsensusStats{
		ItemsWithConsensus:  len(consensusResults),
		RepeatCount:         repeatCount,
		DistributionVariety: make(map[int]int),
	}

	var ratioSum float64
	minRatio := 1.0
	maxRatio := 0.0

	for _, result := range consensusResults {
		ratio := result.Consensus.Ratio
		ratioSum += ratio

		if ratio < minRatio {
			minRatio = ratio
		}
		if ratio > maxRatio {
			maxRatio = ratio
		}

		uniqueAnswers := len(result.Consensus.Distribution)
		stats.DistributionVariety[uniqueAnswers]++
	}

	stats.AvgConsensusRatio = ratioSum / float64(len(consensusResults))
	stats.MinConsensusRatio = minRatio
	stats.MaxConsensusRatio = maxRatio

	return stats
}

func getUniqueClasses(items []string) map[string]bool {
	classSet := make(map[string]bool)
	for _, item := range items {
		classSet[item] = true
	}
	return classSet
}

func getClassMetrics(class string, predictions, actuals []string) (tp, fp, fn int) {
	for i := 0; i < len(predictions); i++ {
		if actuals[i] == class && predictions[i] == class {
			tp++
		} else if actuals[i] != class && predictions[i] == class {
			fp++
		} else if actuals[i] == class && predictions[i] != class {
			fn++
		}
	}
	return tp, fp, fn
}

func countClassOccurrences(class string, items []string) int {
	count := 0
	for _, item := range items {
		if item == class {
			count++
		}
	}
	return count
}

// Helper methods for writing CSV format
func (r *Reporter) writeCSVSummary(writer *csv.Writer) error {
	// Write header for summary section
	if err := writer.Write([]string{"Section", "Metric", "Value"}); err != nil {
		return err
	}

	records := [][]string{
		{"Summary", "Total Items", fmt.Sprintf("%d", r.stats.Total)},
		{"Summary", "Success Rate", fmt.Sprintf("%.2f%%", r.stats.SuccessRate)},
		{"Summary", "Accuracy", fmt.Sprintf("%.2f%%", r.stats.Accuracy)},
		{"Summary", "Balanced Accuracy", fmt.Sprintf("%.2f%%", r.stats.BalancedAccuracy)},
		{"Summary", "Macro F1", fmt.Sprintf("%.2f%%", r.stats.MacroF1)},
		{"Summary", "Micro F1", fmt.Sprintf("%.2f%%", r.stats.MicroF1)},
		{"Summary", "Weighted F1", fmt.Sprintf("%.2f%%", r.stats.WeightedF1)},
		{"Summary", "Cohen's Kappa", fmt.Sprintf("%.4f", r.stats.Kappa)},
		{"Summary", "MCC", fmt.Sprintf("%.4f", r.stats.MCC)},
		{"Performance", "Throughput Per Second", fmt.Sprintf("%.2f", r.stats.ThroughputPerSecond)},
		{"Performance", "Average Response Time", r.stats.AvgTime.String()},
		{"Performance", "P95 Response Time", r.stats.P95Time.String()},
	}

	return writer.WriteAll(records)
}

func (r *Reporter) writeCSVPerClassMetrics(writer *csv.Writer) error {
	// Add separator
	if err := writer.Write([]string{"", "", ""}); err != nil {
		return err
	}
	
	// Write header for per-class metrics
	if err := writer.Write([]string{"Class", "Precision", "Recall", "F1-Score", "Specificity", "NPV", "Jaccard", "Support"}); err != nil {
		return err
	}

	for class := range r.stats.ClassDistribution {
		support := r.stats.ClassDistribution[class]
		record := []string{
			class,
			fmt.Sprintf("%.2f%%", r.stats.Precision[class]*100),
			fmt.Sprintf("%.2f%%", r.stats.Recall[class]*100),
			fmt.Sprintf("%.2f%%", r.stats.F1Score[class]*100),
			fmt.Sprintf("%.2f%%", r.stats.Specificity[class]*100),
			fmt.Sprintf("%.2f%%", r.stats.NPV[class]*100),
			fmt.Sprintf("%.2f%%", r.stats.JaccardScore[class]*100),
			fmt.Sprintf("%d", support),
		}
		if err := writer.Write(record); err != nil {
			return err
		}
	}

	return nil
}

func (r *Reporter) writeCSVHypothesisTests(writer *csv.Writer) error {
	if r.stats.HypothesisTests == nil {
		return nil
	}

	// Add separator
	if err := writer.Write([]string{"", "", ""}); err != nil {
		return err
	}

	// Write header for hypothesis tests
	if err := writer.Write([]string{"Test", "Statistic", "P-Value", "Significant"}); err != nil {
		return err
	}

	if r.stats.HypothesisTests.ChiSquareTest != nil {
		test := r.stats.HypothesisTests.ChiSquareTest
		record := []string{
			"Chi-Square",
			fmt.Sprintf("%.4f", test.Statistic),
			fmt.Sprintf("%.6f", test.PValue),
			fmt.Sprintf("%t", test.Significant),
		}
		if err := writer.Write(record); err != nil {
			return err
		}
	}

	return nil
}

// Helper methods for writing Excel format
func (r *Reporter) writeExcelSummary(f *excelize.File, sheet string) error {
	// Set headers
	headers := []string{"Metric", "Value"}
	for i, header := range headers {
		cell := fmt.Sprintf("%c1", 'A'+i)
		f.SetCellValue(sheet, cell, header)
	}

	// Set data
	data := [][]interface{}{
		{"Total Items", r.stats.Total},
		{"Success Rate", fmt.Sprintf("%.2f%%", r.stats.SuccessRate)},
		{"Accuracy", fmt.Sprintf("%.2f%%", r.stats.Accuracy)},
		{"Balanced Accuracy", fmt.Sprintf("%.2f%%", r.stats.BalancedAccuracy)},
		{"Macro F1", fmt.Sprintf("%.2f%%", r.stats.MacroF1)},
		{"Micro F1", fmt.Sprintf("%.2f%%", r.stats.MicroF1)},
		{"Weighted F1", fmt.Sprintf("%.2f%%", r.stats.WeightedF1)},
		{"Cohen's Kappa", r.stats.Kappa},
		{"MCC", r.stats.MCC},
		{"Throughput/sec", r.stats.ThroughputPerSecond},
	}

	for i, row := range data {
		for j, value := range row {
			cell := fmt.Sprintf("%c%d", 'A'+j, i+2)
			f.SetCellValue(sheet, cell, value)
		}
	}

	return nil
}

func (r *Reporter) writeExcelPerClass(f *excelize.File, sheet string) error {
	// Set headers
	headers := []string{"Class", "Precision", "Recall", "F1-Score", "Specificity", "NPV", "Jaccard", "Support"}
	for i, header := range headers {
		cell := fmt.Sprintf("%c1", 'A'+i)
		f.SetCellValue(sheet, cell, header)
	}

	row := 2
	for class := range r.stats.ClassDistribution {
		support := r.stats.ClassDistribution[class]
		data := []interface{}{
			class,
			r.stats.Precision[class],
			r.stats.Recall[class],
			r.stats.F1Score[class],
			r.stats.Specificity[class],
			r.stats.NPV[class],
			r.stats.JaccardScore[class],
			support,
		}

		for j, value := range data {
			cell := fmt.Sprintf("%c%d", 'A'+j, row)
			f.SetCellValue(sheet, cell, value)
		}
		row++
	}

	return nil
}

func (r *Reporter) writeExcelHypothesis(f *excelize.File, sheet string) error {
	if r.stats.HypothesisTests == nil {
		return nil
	}

	// Set headers
	headers := []string{"Test", "Statistic", "P-Value", "Significant", "Details"}
	for i, header := range headers {
		cell := fmt.Sprintf("%c1", 'A'+i)
		f.SetCellValue(sheet, cell, header)
	}

	row := 2
	if r.stats.HypothesisTests.ChiSquareTest != nil {
		test := r.stats.HypothesisTests.ChiSquareTest
		data := []interface{}{
			"Chi-Square",
			test.Statistic,
			test.PValue,
			test.Significant,
			fmt.Sprintf("df=%d", test.DegreesOfFreedom),
		}

		for j, value := range data {
			cell := fmt.Sprintf("%c%d", 'A'+j, row)
			f.SetCellValue(sheet, cell, value)
		}
		row++
	}

	return nil
}

func (r *Reporter) writeExcelConfusion(f *excelize.File, sheet string) error {
	if len(r.stats.ConfusionMatrix) == 0 {
		return nil
	}

	// Get sorted classes
	var classes []string
	for class := range r.stats.ConfusionMatrix {
		classes = append(classes, class)
	}
	sort.Strings(classes)

	// Set headers
	f.SetCellValue(sheet, "A1", "Actual\\Predicted")
	for i, class := range classes {
		cell := fmt.Sprintf("%c1", 'B'+i)
		f.SetCellValue(sheet, cell, class)
	}

	// Set confusion matrix data
	for i, actualClass := range classes {
		row := i + 2
		f.SetCellValue(sheet, fmt.Sprintf("A%d", row), actualClass)
		
		for j, predictedClass := range classes {
			cell := fmt.Sprintf("%c%d", 'B'+j, row)
			count := r.stats.ConfusionMatrix[actualClass][predictedClass]
			f.SetCellValue(sheet, cell, count)
		}
	}

	return nil
}
