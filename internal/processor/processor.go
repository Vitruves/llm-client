package processor

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"llm-client/internal/client"
	"llm-client/internal/loader"
	"llm-client/internal/logger"
	"llm-client/internal/metrics"
	"llm-client/internal/models"
	"llm-client/internal/parser"
	"llm-client/internal/progress"
	"llm-client/internal/writer"

	"github.com/parquet-go/parquet-go"
	"github.com/xuri/excelize/v2"
)

type Processor struct {
	config         *models.Config
	client         models.Client
	parser         *parser.Parser
	metricsCalc    *metrics.Calculator
	cancelRequests int32
	configFile     string
	inputFile      string
}

type Options struct {
	Limit        int
	ShowProgress bool
	Verbose      bool
	ResumeFile   string
}

func New(config *models.Config) *Processor {
	proc := &Processor{
		config: config,
		parser: parser.New(config.Classification.Parsing),
	}

	if config.Processing.LiveMetrics != nil && config.Processing.LiveMetrics.Enabled {
		// Use classes from config, fallback to parsing Find field, then default
		classes := config.Processing.LiveMetrics.Classes
		if len(classes) == 0 && len(config.Classification.Parsing.Find) > 0 {
			classes = config.Classification.Parsing.Find
		}
		if len(classes) == 0 {
			classes = []string{"0", "1", "2"} // Final fallback for backward compatibility
		}
		proc.metricsCalc = metrics.NewCalculator(config.Processing.LiveMetrics, classes)
	}

	return proc
}

func (p *Processor) SetConfigFile(configFile string) {
	p.configFile = configFile
}

func (p *Processor) ProcessFile(ctx context.Context, inputFile string, opts Options) error {
	// Store input file for resume functionality
	p.inputFile = inputFile
	var data []models.DataRow
	var err error

	// Check if we're resuming from a previous state
	if opts.ResumeFile != "" {
		return p.resumeProcessing(ctx, opts)
	}

	// Initialize client
	p.client, err = client.NewClient(p.config)
	if err != nil {
		return fmt.Errorf("failed to create client: %w", err)
	}
	defer p.client.Close()

	if err := p.client.HealthCheck(ctx); err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}

	data, err = loader.LoadData(inputFile)
	if err != nil {
		return fmt.Errorf("failed to load data: %w", err)
	}

	if opts.Limit > 0 && opts.Limit < len(data) {
		data = data[:opts.Limit]
	}

	p.printProcessingInfo(data)

	// Choose processing method based on streaming configuration
	var results []models.Result
	if p.config.Output.StreamOutput {
		results = p.processWithWorkersStreaming(ctx, data, opts)
	} else {
		results = p.processWithWorkersSimple(ctx, data, opts)
	}

	if atomic.LoadInt32(&p.cancelRequests) > 0 {
		resumeFile := p.savePartialResults(data, results, opts)
		logger.Warning("Processing cancelled. Processed %d items before cancellation.", len(results))
		if resumeFile != "" {
			if p.configFile != "" {
				logger.Info("Resume with: ./llm-client run -c %s --resume %s", p.configFile, resumeFile)
			} else {
				logger.Info("Resume with: ./llm-client run --resume %s", resumeFile)
			}
		}
		return fmt.Errorf("processing cancelled")
	}

	// If streaming is enabled, results are already written, so only save for resume capability
	if p.config.Output.StreamOutput {
		logger.Success("Processing completed successfully. Results written via streaming.")
		// Save config and server info even for streaming output
		if err := p.saveConfigAndServerInfo(); err != nil {
			logger.Warning("Failed to save config and server info: %v", err)
		}
		return nil
	}

	if err := p.saveResults(results); err != nil {
		return err
	}

	// Save client configuration and server info to text file
	if err := p.saveConfigAndServerInfo(); err != nil {
		logger.Warning("Failed to save config and server info: %v", err)
	}

	return nil
}

func (p *Processor) savePartialResults(allData []models.DataRow, results []models.Result, opts Options) string {
	if len(results) == 0 {
		return ""
	}

	// Create map of processed items
	processedItems := make(map[int]bool)
	for _, result := range results {
		processedItems[result.Index] = true
	}

	// Build resume state
	resumeState := models.ResumeState{
		ConfigFile:      p.configFile,
		InputFile:       p.inputFile,
		OutputDirectory: p.config.Output.Directory,
		ProcessedItems:  make([]int, 0, len(processedItems)),
		CompletedCount:  len(results),
		TotalCount:      len(allData),
		Results:         results,
		Timestamp:       time.Now(),
		Options: models.ResumeOptions{
			Workers:      p.config.Processing.Workers,
			Repeat:       p.config.Processing.Repeat,
			Limit:        opts.Limit,
			ShowProgress: opts.ShowProgress,
			Verbose:      opts.Verbose,
		},
	}

	// Convert map to slice
	for idx := range processedItems {
		resumeState.ProcessedItems = append(resumeState.ProcessedItems, idx)
	}

	timestamp := time.Now().Format("20060102_150405")
	resumeFile := filepath.Join(p.config.Output.Directory, fmt.Sprintf("resume_cancelled_%s.json", timestamp))

	if err := os.MkdirAll(p.config.Output.Directory, 0755); err != nil {
		logger.Warning("Failed to create output directory: %v", err)
		resumeFile = fmt.Sprintf("resume_cancelled_%s.json", timestamp)
	}

	file, err := os.Create(resumeFile)
	if err != nil {
		logger.Warning("Failed to create resume file: %v", err)
		return ""
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(resumeState); err != nil {
		logger.Warning("Failed to save resume state: %v", err)
		return ""
	}

	return resumeFile
}

func (p *Processor) resumeProcessing(ctx context.Context, opts Options) error {
	resumeState, err := p.loadResumeState(opts.ResumeFile)
	if err != nil {
		return fmt.Errorf("failed to load resume state: %w", err)
	}

	// Initialize client
	p.client, err = client.NewClient(p.config)
	if err != nil {
		return fmt.Errorf("failed to create client: %w", err)
	}
	defer p.client.Close()

	if err := p.client.HealthCheck(ctx); err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}

	logger.Info("Resuming processing from: %s", resumeState.InputFile)
	logger.Info("Previously completed: %d/%d items", resumeState.CompletedCount, resumeState.TotalCount)

	// Load original data
	data, err := loader.LoadData(resumeState.InputFile)
	if err != nil {
		return fmt.Errorf("failed to load original data: %w", err)
	}

	// Create map of processed items
	processedItems := make(map[int]bool)
	for _, idx := range resumeState.ProcessedItems {
		processedItems[idx] = true
	}

	// Filter out already processed items
	var remainingData []models.DataRow
	for _, row := range data {
		if !processedItems[row.Index] {
			remainingData = append(remainingData, row)
		}
	}

	if len(remainingData) == 0 {
		logger.Info("All items already processed. Saving final results...")
		return p.saveResults(resumeState.Results)
	}

	logger.Info("Resuming with %d remaining items", len(remainingData))

	// Continue processing
	newResults := p.processWithWorkersWithResume(ctx, remainingData, opts, processedItems)

	// Merge results
	allResults := append(resumeState.Results, newResults...)

	if atomic.LoadInt32(&p.cancelRequests) > 0 {
		logger.Warning("Processing cancelled again. Processed %d additional items.", len(newResults))
		logger.Info("Updating resume state...")

		// Update resume state
		for _, result := range newResults {
			processedItems[result.Index] = true
		}

		resumeFile := p.saveResumeState(resumeState.InputFile, data, allResults, processedItems, opts)
		logger.Info("To resume processing, run:")
		logger.Info("./llm-client run --resume %s", resumeFile)

		return fmt.Errorf("processing cancelled")
	}

	// Clean up resume file on successful completion
	os.Remove(opts.ResumeFile)
	logger.Success("Processing completed successfully. Resume file cleaned up.")

	return p.saveResults(allResults)
}

func (p *Processor) saveResumeState(inputFile string, allData []models.DataRow, results []models.Result, processedItems map[int]bool, opts Options) string {
	processedList := make([]int, 0, len(processedItems))
	for idx := range processedItems {
		processedList = append(processedList, idx)
	}

	resumeState := models.ResumeState{
		ConfigFile:      "", // Will be set by caller
		InputFile:       inputFile,
		OutputDirectory: p.config.Output.Directory,
		ProcessedItems:  processedList,
		CompletedCount:  len(results),
		TotalCount:      len(allData),
		Results:         results,
		Timestamp:       time.Now(),
		Options: models.ResumeOptions{
			Workers:      p.config.Processing.Workers,
			Repeat:       p.config.Processing.Repeat,
			Limit:        opts.Limit,
			ShowProgress: opts.ShowProgress,
			Verbose:      opts.Verbose,
		},
	}

	timestamp := time.Now().Format("20060102_150405")
	resumeFile := filepath.Join(p.config.Output.Directory, fmt.Sprintf("resume_%s.json", timestamp))

	if err := os.MkdirAll(p.config.Output.Directory, 0755); err != nil {
		logger.Warning("Failed to create output directory: %v", err)
		resumeFile = fmt.Sprintf("resume_%s.json", timestamp)
	}

	file, err := os.Create(resumeFile)
	if err != nil {
		logger.Warning("Failed to create resume file: %v", err)
		return resumeFile
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(resumeState); err != nil {
		logger.Warning("Failed to save resume state: %v", err)
	}

	return resumeFile
}

func (p *Processor) loadResumeState(resumeFile string) (*models.ResumeState, error) {
	data, err := os.ReadFile(resumeFile)
	if err != nil {
		return nil, err
	}

	var resumeState models.ResumeState
	if err := json.Unmarshal(data, &resumeState); err != nil {
		return nil, err
	}

	return &resumeState, nil
}

func (p *Processor) printProcessingInfo(data []models.DataRow) {
	logger.Info("Starting processing: %d items, %d workers", len(data), p.config.Processing.Workers)
	if p.config.Processing.Repeat > 1 {
		logger.Info("Consensus mode: %d requests per item", p.config.Processing.Repeat)
	}
	if p.metricsCalc != nil {
		logger.Info("Live metrics: %s", p.metricsCalc.GetMetricName())
	}
}

func (p *Processor) calculateOptimalBufferSizes(dataLen int) (resultBuffer, dataBuffer int) {
	workers := p.config.Processing.Workers

	resultBuffer = workers * 4
	if resultBuffer > dataLen {
		resultBuffer = dataLen
	}
	if resultBuffer < 10 {
		resultBuffer = 10
	}

	dataBuffer = workers * 2
	if dataBuffer > 1000 {
		dataBuffer = 1000
	}
	if dataBuffer < 10 {
		dataBuffer = 10
	}

	return resultBuffer, dataBuffer
}

func (p *Processor) processWithWorkersSimple(ctx context.Context, data []models.DataRow, opts Options) []models.Result {
	resultBuffer, dataBuffer := p.calculateOptimalBufferSizes(len(data))
	resultsCh := make(chan models.Result, resultBuffer)
	dataCh := make(chan models.DataRow, dataBuffer)

	var processed, succeeded, failed int64
	var prog *progress.Progress

	if opts.ShowProgress {
		prog = progress.NewWithMetrics(len(data), p.metricsCalc)
		prog.Start()
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	go p.dataFeeder(ctx, dataCh, data)

	var wg sync.WaitGroup

	// Simple workers - no resume mechanism overhead
	for i := 0; i < p.config.Processing.Workers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()

			for {
				select {
				case <-ctx.Done():
					atomic.StoreInt32(&p.cancelRequests, 1)
					return
				case row, ok := <-dataCh:
					if !ok {
						return
					}

					if atomic.LoadInt32(&p.cancelRequests) > 0 {
						return
					}

					result := p.processRow(ctx, row, opts.Verbose)

					select {
					case resultsCh <- result:
						p.updateCounters(result, &processed, &succeeded, &failed)
						p.logVerboseResult(workerID, row, result, opts.Verbose, atomic.LoadInt64(&processed))
						p.updateProgress(prog, &processed, &succeeded, &failed)
					case <-ctx.Done():
						atomic.StoreInt32(&p.cancelRequests, 1)
						return
					}
				}
			}
		}(i)
	}

	go func() {
		wg.Wait()
		close(resultsCh)
		if prog != nil {
			prog.Stop()
		}
	}()

	return p.collectResults(resultsCh)
}

func (p *Processor) processWithWorkersStreaming(ctx context.Context, data []models.DataRow, opts Options) []models.Result {
	var streamWriter writer.StreamWriter
	var err error

	// Create stream writer if enabled
	if p.config.Output.StreamOutput {
		timestamp := time.Now().Format("20060102_150405")
		streamWriter, err = writer.NewStreamWriter(p.config.Output.Format, p.config.Output.Directory, timestamp, &p.config.Output)
		if err != nil {
			logger.Warning("Failed to create stream writer: %v, falling back to batch mode", err)
		} else {
			defer func() {
				if err := streamWriter.Close(); err != nil {
					logger.Warning("Failed to close stream writer: %v", err)
				} else if streamWriter != nil {
					logger.Info("Results streamed to: %s", streamWriter.GetFilename())
				}
			}()
		}
	}

	resultBuffer, dataBuffer := p.calculateOptimalBufferSizes(len(data))
	resultsCh := make(chan models.Result, resultBuffer)
	dataCh := make(chan models.DataRow, dataBuffer)

	var processed, succeeded, failed int64
	var prog *progress.Progress

	if opts.ShowProgress {
		prog = progress.NewWithMetrics(len(data), p.metricsCalc)
		prog.Start()
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	go p.dataFeeder(ctx, dataCh, data)

	var wg sync.WaitGroup

	// Workers with streaming support
	for i := 0; i < p.config.Processing.Workers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()

			for {
				select {
				case <-ctx.Done():
					atomic.StoreInt32(&p.cancelRequests, 1)
					return
				case row, ok := <-dataCh:
					if !ok {
						return
					}

					if atomic.LoadInt32(&p.cancelRequests) > 0 {
						return
					}

					result := p.processRow(ctx, row, opts.Verbose)

					select {
					case resultsCh <- result:
						p.updateCounters(result, &processed, &succeeded, &failed)
						p.logVerboseResult(workerID, row, result, opts.Verbose, atomic.LoadInt64(&processed))
						p.updateProgress(prog, &processed, &succeeded, &failed)
					case <-ctx.Done():
						atomic.StoreInt32(&p.cancelRequests, 1)
						return
					}
				}
			}
		}(i)
	}

	go func() {
		wg.Wait()
		close(resultsCh)
		if prog != nil {
			prog.Stop()
		}
	}()

	return p.collectResultsWithStreaming(resultsCh, streamWriter)
}

func (p *Processor) processWithWorkersWithResume(ctx context.Context, data []models.DataRow, opts Options, processedItems map[int]bool) []models.Result {
	resultBuffer, dataBuffer := p.calculateOptimalBufferSizes(len(data))
	resultsCh := make(chan models.Result, resultBuffer)
	dataCh := make(chan models.DataRow, dataBuffer)

	var processed, succeeded, failed int64
	var prog *progress.Progress

	if opts.ShowProgress {
		prog = progress.NewWithMetrics(len(data), p.metricsCalc)
		prog.Start()
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	go p.dataFeeder(ctx, dataCh, data)

	var wg sync.WaitGroup

	// Use complex workers for resume processing
	for i := 0; i < p.config.Processing.Workers; i++ {
		wg.Add(1)
		go p.workerWithResume(ctx, i, dataCh, resultsCh, &wg, &processed, &succeeded, &failed, prog, opts.Verbose, processedItems)
	}

	go func() {
		wg.Wait()
		close(resultsCh)
		if prog != nil {
			prog.Stop()
		}
	}()

	return p.collectResults(resultsCh)
}

func (p *Processor) dataFeeder(ctx context.Context, dataCh chan<- models.DataRow, data []models.DataRow) {
	defer close(dataCh)
	for _, row := range data {
		select {
		case dataCh <- row:
		case <-ctx.Done():
			return
		}
	}
}

func (p *Processor) workerWithResume(ctx context.Context, workerID int, dataCh <-chan models.DataRow,
	resultsCh chan<- models.Result, wg *sync.WaitGroup,
	processed, succeeded, failed *int64, prog *progress.Progress, verbose bool,
	processedItems map[int]bool) {

	defer wg.Done()

	for {
		select {
		case <-ctx.Done():
			atomic.StoreInt32(&p.cancelRequests, 1)
			return
		case row, ok := <-dataCh:
			if !ok {
				return
			}

			if atomic.LoadInt32(&p.cancelRequests) > 0 {
				return
			}

			result := p.processRow(ctx, row, verbose)

			select {
			case resultsCh <- result:
				processedItems[result.Index] = true
				p.updateCounters(result, processed, succeeded, failed)
				p.logVerboseResult(workerID, row, result, verbose, atomic.LoadInt64(processed))
				p.updateProgress(prog, processed, succeeded, failed)

			case <-ctx.Done():
				atomic.StoreInt32(&p.cancelRequests, 1)
				return
			}
		}
	}
}

func (p *Processor) updateCounters(result models.Result, processed, succeeded, failed *int64) {
	atomic.AddInt64(processed, 1)
	if result.Success {
		atomic.AddInt64(succeeded, 1)
	} else {
		atomic.AddInt64(failed, 1)
	}
}

func (p *Processor) logVerboseResult(workerID int, row models.DataRow, result models.Result, verbose bool, current int64) {
	if !verbose || current > 10 {
		return
	}

	consensusInfo := ""
	if result.Consensus != nil {
		consensusInfo = fmt.Sprintf(" [consensus: %s %.1f%%]",
			result.Consensus.FinalAnswer, result.Consensus.Ratio*100)
	}

	logger.Debug("Worker %d completed row %d: %s (%.1fms)%s",
		workerID, row.Index, result.FinalAnswer,
		float64(result.ResponseTime.Nanoseconds())/1e6, consensusInfo)
}

func (p *Processor) updateProgress(prog *progress.Progress, processed, succeeded, failed *int64) {
	if prog != nil {
		prog.Update(int(atomic.LoadInt64(processed)),
			int(atomic.LoadInt64(succeeded)),
			int(atomic.LoadInt64(failed)))
	}
}

func (p *Processor) collectResults(resultsCh <-chan models.Result) []models.Result {
	var allResults []models.Result
	for result := range resultsCh {
		allResults = append(allResults, result)
		if atomic.LoadInt32(&p.cancelRequests) > 0 {
			break
		}
	}
	return allResults
}

func (p *Processor) collectResultsWithStreaming(resultsCh <-chan models.Result, streamWriter writer.StreamWriter) []models.Result {
	var allResults []models.Result
	saveEvery := p.config.Output.StreamSaveEvery
	if saveEvery <= 0 {
		saveEvery = 1 // Default to flush every result
	}

	for result := range resultsCh {
		// Also collect for potential resume state first
		allResults = append(allResults, result)

		// Stream write the result immediately
		if streamWriter != nil {
			if err := streamWriter.WriteResult(result); err != nil {
				logger.Warning("Failed to stream write result %d: %v", result.Index, err)
			}

			// Force flush every N results
			if len(allResults)%saveEvery == 0 {
				if err := streamWriter.Flush(); err != nil {
					logger.Warning("Failed to flush stream writer: %v", err)
				}
			}
		}

		if atomic.LoadInt32(&p.cancelRequests) > 0 {
			break
		}
	}
	return allResults
}

func (p *Processor) processRow(ctx context.Context, row models.DataRow, verbose bool) models.Result {
	if atomic.LoadInt32(&p.cancelRequests) > 0 {
		return models.Result{
			Index:     row.Index,
			InputText: row.Text,
			Success:   false,
			Error:     "cancelled",
		}
	}

	if p.config.Processing.Repeat <= 1 {
		return p.processSingle(ctx, row, verbose)
	}
	return p.processMultiple(ctx, row, verbose)
}

func (p *Processor) processSingle(ctx context.Context, row models.DataRow, verbose bool) models.Result {
	start := time.Now()
	groundTruth := p.extractGroundTruth(row)

	if verbose {
		logger.Debug("Processing row %d: %s", row.Index, p.truncate(p.extractInputText(row), 80))
	}

	req := p.buildRequest(row)
	resp, err := p.client.SendRequest(ctx, req)
	responseTime := time.Since(start)

	if err != nil {
		if verbose {
			logger.Debug("Row %d failed: %v", row.Index, err)
		}
		return p.createErrorResult(row, groundTruth, err.Error(), "", responseTime)
	}

	if !resp.Success {
		if verbose {
			logger.Debug("Row %d error: %s", row.Index, resp.Error)
		}
		return p.createErrorResult(row, groundTruth, resp.Error, resp.Content, responseTime)
	}

	return p.createSuccessResult(row, groundTruth, resp.Content, responseTime, verbose)
}

func (p *Processor) processMultiple(ctx context.Context, row models.DataRow, verbose bool) models.Result {
	start := time.Now()
	groundTruth := p.extractGroundTruth(row)

	if verbose {
		logger.Debug("Processing row %d with %d attempts: %s",
			row.Index, p.config.Processing.Repeat, p.truncate(p.extractInputText(row), 80))
	}

	req := p.buildRequest(row)
	attempts, answers := p.executeAttempts(ctx, req, verbose)

	result := models.Result{
		Index:        row.Index,
		InputText:    p.extractInputText(row),
		OriginalData: row.Data,
		GroundTruth:  groundTruth,
		ResponseTime: time.Since(start),
		Attempts:     attempts,
	}

	if len(answers) == 0 {
		result.Success = false
		result.Error = "all attempts failed"
		return result
	}

	p.finalizeMultipleResult(&result, attempts, answers, verbose)
	return result
}

func (p *Processor) executeAttempts(ctx context.Context, req models.Request, verbose bool) ([]models.Attempt, []string) {
	attempts := make([]models.Attempt, 0, p.config.Processing.Repeat)
	answers := make([]string, 0, p.config.Processing.Repeat)

	for i := 0; i < p.config.Processing.Repeat; i++ {
		select {
		case <-ctx.Done():
			return attempts, answers
		default:
		}

		if atomic.LoadInt32(&p.cancelRequests) > 0 {
			return attempts, answers
		}

		if i > 0 && p.config.Processing.RateLimit {
			time.Sleep(time.Second)
		}

		attempt := p.executeAttempt(ctx, req)
		attempts = append(attempts, attempt)

		if attempt.Success {
			answers = append(answers, attempt.Answer)
		}

		if verbose {
			status := attempt.Answer
			if !attempt.Success {
				status = "failed"
			}
			logger.Debug("Attempt %d/%d: %s", i+1, p.config.Processing.Repeat, status)
		}
	}

	return attempts, answers
}

func (p *Processor) executeAttempt(ctx context.Context, req models.Request) models.Attempt {
	attemptStart := time.Now()
	resp, err := p.client.SendRequest(ctx, req)

	attempt := models.Attempt{
		ResponseTime: time.Since(attemptStart),
		Success:      false,
	}

	if err != nil {
		attempt.Error = err.Error()
		return attempt
	}

	if !resp.Success {
		attempt.Error = resp.Error
		attempt.Response = resp.Content
		return attempt
	}

	attempt.Success = true
	attempt.Response = resp.Content

	// In minimal mode, skip parsing and use raw response
	if p.config.Processing.MinimalMode {
		attempt.Answer = resp.Content
	} else {
		finalAnswer, _ := p.parser.ParseWithThinking(resp.Content)
		attempt.Answer = metrics.NormalizeLabel(finalAnswer)
	}

	return attempt
}

func (p *Processor) finalizeMultipleResult(result *models.Result, attempts []models.Attempt, answers []string, verbose bool) {
	var consensus *models.Consensus

	// In minimal mode, use first successful response without consensus calculation
	if p.config.Processing.MinimalMode {
		result.Success = true
		result.FinalAnswer = attempts[0].Response
		result.RawResponse = attempts[0].Response
		result.ThinkingContent = ""

		// Simple consensus in minimal mode
		consensus = &models.Consensus{
			FinalAnswer:  attempts[0].Response,
			Count:        1,
			Total:        len(attempts),
			Ratio:        1.0,
			Distribution: map[string]int{attempts[0].Response: 1},
		}
	} else {
		consensus = p.calculateConsensus(answers)
		result.Success = true
		result.FinalAnswer = consensus.FinalAnswer
		result.RawResponse = attempts[0].Response

		if len(attempts) > 0 && attempts[0].Success {
			_, thinkingContent := p.parser.ParseWithThinking(attempts[0].Response)
			result.ThinkingContent = thinkingContent
		}
	}

	result.Consensus = consensus

	if verbose {
		if p.config.Processing.MinimalMode {
			logger.Debug("[MINIMAL] Row %d completed with %d attempts",
				result.Index, len(attempts))
		} else {
			logger.Debug("Row %d consensus: %s (%.1f%% agreement, %d/%d successful)",
				result.Index, consensus.FinalAnswer, consensus.Ratio*100, len(answers), p.config.Processing.Repeat)

			if len(consensus.Distribution) > 1 {
				logger.Debug("   Distribution: %v", consensus.Distribution)
			}
		}
	}

	// Skip live metrics in minimal mode
	if !p.config.Processing.MinimalMode {
		p.updateLiveMetrics(*result)
	}
}

func (p *Processor) buildRequest(row models.DataRow) models.Request {
	return models.Request{
		Messages: []models.Message{
			models.NewTextMessage("system", p.config.Classification.Template.System),
			models.NewTextMessage("user", p.applyTemplate(row)),
		},
		Options: p.config.Model.Parameters,
	}
}

func (p *Processor) createErrorResult(row models.DataRow, groundTruth, errorMsg, rawResponse string, responseTime time.Duration) models.Result {
	return models.Result{
		Index:        row.Index,
		InputText:    p.extractInputText(row),
		OriginalData: row.Data,
		GroundTruth:  groundTruth,
		Success:      false,
		Error:        errorMsg,
		RawResponse:  rawResponse,
		ResponseTime: responseTime,
	}
}

func (p *Processor) createSuccessResult(row models.DataRow, groundTruth, rawResponse string, responseTime time.Duration, verbose bool) models.Result {
	var finalAnswer, thinkingContent string
	var normalizedAnswer string

	// In minimal mode, skip parsing and normalization
	if p.config.Processing.MinimalMode {
		finalAnswer = rawResponse
		normalizedAnswer = rawResponse
		thinkingContent = ""
	} else {
		finalAnswer, thinkingContent = p.parser.ParseWithThinking(rawResponse)
		normalizedAnswer = metrics.NormalizeLabel(finalAnswer)
	}

	if verbose {
		status := "OK"
		if !p.config.Processing.MinimalMode && normalizedAnswer != groundTruth {
			status = "WRONG"
		}
		if p.config.Processing.MinimalMode {
			logger.Debug("[MINIMAL] Row %d completed (%.2fms)",
				row.Index, float64(responseTime.Nanoseconds())/1e6)
		} else {
			logger.Debug("[%s] Row %d completed: %s (GT: %s) (%.2fms)",
				status, row.Index, normalizedAnswer, groundTruth, float64(responseTime.Nanoseconds())/1e6)
			if thinkingContent != "" {
				logger.Debug("Row %d thinking: %s", row.Index, p.truncate(thinkingContent, 100))
			}
		}
	}

	result := models.Result{
		Index:           row.Index,
		InputText:       p.extractInputText(row),
		OriginalData:    row.Data,
		GroundTruth:     groundTruth,
		FinalAnswer:     normalizedAnswer,
		RawResponse:     rawResponse,
		ThinkingContent: thinkingContent,
		Success:         true,
		ResponseTime:    responseTime,
	}

	// Skip live metrics in minimal mode
	if !p.config.Processing.MinimalMode {
		p.updateLiveMetrics(result)
	}
	return result
}

func (p *Processor) calculateConsensus(answers []string) *models.Consensus {
	distribution := make(map[string]int)
	for _, answer := range answers {
		distribution[answer]++
	}

	var maxCount int
	var finalAnswer string
	for answer, count := range distribution {
		if count > maxCount {
			maxCount = count
			finalAnswer = answer
		}
	}

	return &models.Consensus{
		FinalAnswer:  finalAnswer,
		Count:        maxCount,
		Total:        len(answers),
		Ratio:        float64(maxCount) / float64(len(answers)),
		Distribution: distribution,
	}
}

func (p *Processor) extractGroundTruth(row models.DataRow) string {
	if p.config.Processing.LiveMetrics == nil {
		return ""
	}

	gtField := p.config.Processing.LiveMetrics.GroundTruth
	if gtValue, ok := row.Data[gtField]; ok {
		return metrics.NormalizeLabel(fmt.Sprintf("%v", gtValue))
	}
	return ""
}

func (p *Processor) extractInputText(row models.DataRow) string {
	// Priority 1: Use field mapping from classification config
	if p.config.Classification.FieldMapping != nil && p.config.Classification.FieldMapping.InputTextField != "" {
		fieldName := p.config.Classification.FieldMapping.InputTextField
		if value, ok := row.Data[fieldName]; ok {
			return fmt.Sprintf("%v", value)
		}
	}

	// Priority 2: Use output config (backward compatibility)
	if p.config.Output.InputTextField != "" {
		if value, ok := row.Data[p.config.Output.InputTextField]; ok {
			return fmt.Sprintf("%v", value)
		}
	}

	// Priority 3: Try to find the main content field used in template
	template := p.config.Classification.Template.User

	// Check for common placeholders in order of preference
	placeholders := []string{"REVIEW", "review", "TEXT", "text", "CONTENT", "content", "MESSAGE", "message"}

	for _, placeholder := range placeholders {
		if strings.Contains(template, "{"+placeholder+"}") {
			// Try case variations
			for _, key := range []string{placeholder, strings.ToLower(placeholder), strings.ToUpper(placeholder)} {
				if value, ok := row.Data[key]; ok {
					return fmt.Sprintf("%v", value)
				}
			}
		}
	}

	// Priority 4: Fallback to row.Text (original behavior)
	return row.Text
}

func (p *Processor) updateLiveMetrics(result models.Result) {
	if p.metricsCalc == nil || !result.Success || result.GroundTruth == "" {
		return
	}
	p.metricsCalc.AddResult(result.FinalAnswer, result.GroundTruth)
}

func (p *Processor) applyTemplate(row models.DataRow) string {
	template := p.config.Classification.Template.User
	template = strings.ReplaceAll(template, "{text}", row.Text)
	template = strings.ReplaceAll(template, "{index}", fmt.Sprintf("%d", row.Index))
	return p.processPlaceholders(template, row.Data, "")
}

func (p *Processor) processPlaceholders(template string, data map[string]interface{}, prefix string) string {
	for key, value := range data {
		fullKey := key
		if prefix != "" {
			fullKey = prefix + "." + key
		}

		placeholders := []string{
			fmt.Sprintf("{%s}", key),
			fmt.Sprintf("{%s}", strings.ToUpper(key)),
			fmt.Sprintf("{%s}", strings.ToLower(key)),
			fmt.Sprintf("{%s}", fullKey),
			fmt.Sprintf("{%s}", strings.ToUpper(fullKey)),
			fmt.Sprintf("{%s}", strings.ToLower(fullKey)),
		}

		var formattedValue string
		switch v := value.(type) {
		case string:
			formattedValue = v
		case int, int64, int32:
			formattedValue = fmt.Sprintf("%d", v)
		case float64, float32:
			formattedValue = fmt.Sprintf("%.2f", v)
		case bool:
			formattedValue = fmt.Sprintf("%t", v)
		case []interface{}:
			items := make([]string, len(v))
			for i, item := range v {
				items[i] = fmt.Sprintf("%v", item)
			}
			formattedValue = strings.Join(items, ", ")
		case map[string]interface{}:
			template = p.processPlaceholders(template, v, fullKey)
			formattedValue = p.formatNestedObject(v)
		default:
			formattedValue = fmt.Sprintf("%v", v)
		}

		for _, placeholder := range placeholders {
			template = strings.ReplaceAll(template, placeholder, formattedValue)
		}
	}

	return template
}

func (p *Processor) formatNestedObject(obj map[string]interface{}) string {
	parts := make([]string, 0, len(obj))
	for k, v := range obj {
		parts = append(parts, fmt.Sprintf("%s: %v", k, v))
	}
	return strings.Join(parts, ", ")
}

func (p *Processor) truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}

func (p *Processor) saveResults(results []models.Result) error {
	if err := os.MkdirAll(p.config.Output.Directory, 0755); err != nil {
		return err
	}

	timestamp := time.Now().Format("20060102_150405")

	switch p.config.Output.Format {
	case "json":
		return p.saveAsJSON(results, timestamp)
	case "csv":
		return p.saveAsCSV(results, timestamp)
	case "parquet":
		return p.saveAsParquet(results, timestamp)
	case "xlsx":
		return p.saveAsXLSX(results, timestamp)
	default:
		return fmt.Errorf("unsupported output format: %s", p.config.Output.Format)
	}
}

func (p *Processor) saveAsJSON(results []models.Result, timestamp string) error {
	filename := filepath.Join(p.config.Output.Directory, fmt.Sprintf("results_%s.json", timestamp))

	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Build output data, including original columns
	output := p.buildOutputData(results)

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(output)
}

func (p *Processor) saveAsCSV(results []models.Result, timestamp string) error {
	filename := filepath.Join(p.config.Output.Directory, fmt.Sprintf("results_%s.csv", timestamp))

	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Collect all unique original data keys to form a comprehensive header
	originalHeaders := make(map[string]struct{})
	for _, result := range results {
		for key := range result.OriginalData {
			originalHeaders[key] = struct{}{}
		}
	}

	var sortedOriginalHeaders []string
	for key := range originalHeaders {
		sortedOriginalHeaders = append(sortedOriginalHeaders, key)
	}
	sort.Strings(sortedOriginalHeaders)

	// Write header
	header := []string{"index", "input_text", "ground_truth", "final_answer", "success", "response_time_ms"}
	header = append(header, sortedOriginalHeaders...)

	if p.config.Output.IncludeThinking {
		header = append(header, "thinking_content")
	}
	if p.config.Output.IncludeRawResponse {
		header = append(header, "raw_response")
	}

	if err := writer.Write(header); err != nil {
		return err
	}

	// Write data
	for _, result := range results {
		row := []string{
			strconv.Itoa(result.Index),
			result.InputText,
			result.GroundTruth,
			result.FinalAnswer,
			strconv.FormatBool(result.Success),
			strconv.FormatInt(result.ResponseTime.Nanoseconds()/1000000, 10),
		}

		// Add original data values
		for _, h := range sortedOriginalHeaders {
			val := result.OriginalData[h]
			row = append(row, fmt.Sprintf("%v", val))
		}

		if p.config.Output.IncludeThinking {
			row = append(row, result.ThinkingContent)
		}
		if p.config.Output.IncludeRawResponse {
			row = append(row, result.RawResponse)
		}

		if err := writer.Write(row); err != nil {
			return err
		}
	}

	return nil
}

// ParquetResult represents the structure for parquet output
type ParquetResult struct {
	Index           int32  `parquet:"index"`
	InputText       string `parquet:"input_text"`
	GroundTruth     string `parquet:"ground_truth"`
	FinalAnswer     string `parquet:"final_answer"`
	Success         bool   `parquet:"success"`
	ResponseTimeMs  int64  `parquet:"response_time_ms"`
	ThinkingContent string `parquet:"thinking_content,optional"`
	RawResponse     string `parquet:"raw_response,optional"`
	OriginalData    string `parquet:"original_data,optional"`
}

func (p *Processor) saveAsParquet(results []models.Result, timestamp string) error {
	filename := filepath.Join(p.config.Output.Directory, fmt.Sprintf("results_%s.parquet", timestamp))

	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Convert results to parquet-compatible format
	parquetResults := make([]ParquetResult, len(results))
	for i, result := range results {
		// Convert OriginalData map to JSON string for parquet compatibility
		var originalDataStr string
		if len(result.OriginalData) > 0 {
			if jsonBytes, err := json.Marshal(result.OriginalData); err == nil {
				originalDataStr = string(jsonBytes)
			}
		}

		parquetResults[i] = ParquetResult{
			Index:           int32(result.Index),
			InputText:       result.InputText,
			GroundTruth:     result.GroundTruth,
			FinalAnswer:     result.FinalAnswer,
			Success:         result.Success,
			ResponseTimeMs:  result.ResponseTime.Nanoseconds() / 1000000,
			ThinkingContent: result.ThinkingContent,
			RawResponse:     result.RawResponse,
			OriginalData:    originalDataStr,
		}
	}

	return parquet.Write(file, parquetResults)
}

func (p *Processor) saveAsXLSX(results []models.Result, timestamp string) error {
	filename := filepath.Join(p.config.Output.Directory, fmt.Sprintf("results_%s.xlsx", timestamp))

	f := excelize.NewFile()
	defer f.Close()

	sheetName := "Results"
	f.SetSheetName("Sheet1", sheetName)

	// Collect all unique original data keys to form a comprehensive header
	originalHeaders := make(map[string]struct{})
	for _, result := range results {
		for key := range result.OriginalData {
			originalHeaders[key] = struct{}{}
		}
	}

	var sortedOriginalHeaders []string
	for key := range originalHeaders {
		sortedOriginalHeaders = append(sortedOriginalHeaders, key)
	}
	sort.Strings(sortedOriginalHeaders)

	// Write header
	headers := []string{"Index", "Input Text", "Ground Truth", "Final Answer", "Success", "Response Time (ms)"}
	headers = append(headers, sortedOriginalHeaders...)

	if p.config.Output.IncludeThinking {
		headers = append(headers, "Thinking Content")
	}
	if p.config.Output.IncludeRawResponse {
		headers = append(headers, "Raw Response")
	}

	for i, header := range headers {
		cell := fmt.Sprintf("%c1", 'A'+i)
		f.SetCellValue(sheetName, cell, header)
	}

	// Write data
	for i, result := range results {
		rowNum := i + 2 // Start from row 2 (after header)

		f.SetCellValue(sheetName, fmt.Sprintf("A%d", rowNum), result.Index)
		f.SetCellValue(sheetName, fmt.Sprintf("B%d", rowNum), result.InputText)
		f.SetCellValue(sheetName, fmt.Sprintf("C%d", rowNum), result.GroundTruth)
		f.SetCellValue(sheetName, fmt.Sprintf("D%d", rowNum), result.FinalAnswer)
		f.SetCellValue(sheetName, fmt.Sprintf("E%d", rowNum), result.Success)
		f.SetCellValue(sheetName, fmt.Sprintf("F%d", rowNum), result.ResponseTime.Nanoseconds()/1000000)

		col := 'G'
		// Add original data values dynamically
		for _, h := range sortedOriginalHeaders {
			val := result.OriginalData[h]
			f.SetCellValue(sheetName, fmt.Sprintf("%c%d", col, rowNum), fmt.Sprintf("%v", val))
			col++
		}

		if p.config.Output.IncludeThinking {
			f.SetCellValue(sheetName, fmt.Sprintf("%c%d", col, rowNum), result.ThinkingContent)
			col++
		}
		if p.config.Output.IncludeRawResponse {
			f.SetCellValue(sheetName, fmt.Sprintf("%c%d", col, rowNum), result.RawResponse)
			col++
		}
	}

	return f.SaveAs(filename)
}

func (p *Processor) buildOutputData(results []models.Result) map[string]interface{} {
	summary := p.buildSummary(results)

	outputResults := make([]map[string]interface{}, len(results))
	for i, result := range results {
		outputResult := map[string]interface{}{
			"index":            result.Index,
			"input_text":       result.InputText,
			"ground_truth":     result.GroundTruth,
			"final_answer":     result.FinalAnswer,
			"raw_response":     result.RawResponse,
			"thinking_content": result.ThinkingContent,
			"success":          result.Success,
			"error":            result.Error,
			"response_time_ms": result.ResponseTime.Nanoseconds() / 1000000,
			"attempts":         result.Attempts,
			"consensus":        result.Consensus,
			"tool_calls":       result.ToolCalls,
			"usage":            result.Usage,
		}
		// Merge original data directly into the top level of each result in JSON output
		for k, v := range result.OriginalData {
			outputResult[k] = v
		}
		outputResults[i] = outputResult
	}

	output := map[string]interface{}{
		"results": outputResults,
		"summary": summary,
		"config":  p.config,
	}

	if serverInfo := p.getServerInfo(); serverInfo != nil {
		output["server_info"] = serverInfo
	}

	return output
}

func (p *Processor) buildSummary(results []models.Result) map[string]interface{} {
	successCount := p.countSuccessful(results)
	summary := map[string]interface{}{
		"total":   len(results),
		"success": successCount,
		"failed":  len(results) - successCount,
	}

	if p.metricsCalc != nil {
		summary["live_metrics"] = map[string]interface{}{
			"metric_name":  p.metricsCalc.GetMetricName(),
			"metric_value": p.metricsCalc.GetCurrentMetric(),
		}
	}

	if consensusStats := p.calculateConsensusStats(results); consensusStats != nil {
		summary["consensus_stats"] = consensusStats
	}

	return summary
}

func (p *Processor) getServerInfo() *models.ServerInfo {
	if p.client == nil {
		return nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	info, err := p.client.GetServerInfo(ctx)
	if err != nil {
		return nil
	}
	return info
}

func (p *Processor) countSuccessful(results []models.Result) int {
	count := 0
	for _, result := range results {
		if result.Success {
			count++
		}
	}
	return count
}

func (p *Processor) calculateConsensusStats(results []models.Result) map[string]interface{} {
	if p.config.Processing.Repeat <= 1 {
		return nil
	}

	var totalConsensusRatio float64
	var consensusCount int
	distributionCounts := make(map[int]int)

	for _, result := range results {
		if result.Consensus != nil {
			totalConsensusRatio += result.Consensus.Ratio
			consensusCount++
			distributionCounts[len(result.Consensus.Distribution)]++
		}
	}

	if consensusCount == 0 {
		return nil
	}

	return map[string]interface{}{
		"repeat_count":         p.config.Processing.Repeat,
		"items_with_consensus": consensusCount,
		"avg_consensus_ratio":  totalConsensusRatio / float64(consensusCount),
		"distribution_variety": distributionCounts,
	}
}

func (p *Processor) saveConfigAndServerInfo() error {
	timestamp := time.Now().Format("20060102_150405")
	filename := filepath.Join(p.config.Output.Directory, fmt.Sprintf("config_and_server_info_%s.txt", timestamp))

	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create config file: %w", err)
	}
	defer file.Close()

	// Write header
	fmt.Fprintf(file, "LLM Client Configuration and Server Information\n")
	fmt.Fprintf(file, "Generated: %s\n", time.Now().Format("2006-01-02 15:04:05"))

	// Write configuration file path if available
	if p.configFile != "" {
		fmt.Fprintf(file, "Configuration File: %s\n", p.configFile)
	}
	if p.inputFile != "" {
		fmt.Fprintf(file, "Input File: %s\n", p.inputFile)
	}
	fmt.Fprintf(file, "\n")

	// Write provider configuration
	fmt.Fprintf(file, "PROVIDER CONFIGURATION:\n")
	fmt.Fprintf(file, "  Provider: %s\n", p.config.Provider.Name)
	fmt.Fprintf(file, "  Base URL: %s\n", p.config.Provider.BaseURL)
	fmt.Fprintf(file, "  Timeout: %s\n", p.config.Provider.Timeout)
	if p.config.Provider.APIKey != "" {
		fmt.Fprintf(file, "  API Key: [REDACTED]\n")
	}
	fmt.Fprintf(file, "\n")

	// Write model configuration
	fmt.Fprintf(file, "MODEL CONFIGURATION:\n")
	fmt.Fprintf(file, "  Model Name: %s\n", p.config.Model.Name)

	// Write model parameters
	params := p.config.Model.Parameters
	fmt.Fprintf(file, "  Parameters:\n")

	if params.Temperature != nil {
		fmt.Fprintf(file, "    Temperature: %v\n", *params.Temperature)
	}
	if params.MaxTokens != nil {
		fmt.Fprintf(file, "    Max Tokens: %v\n", *params.MaxTokens)
	}
	if params.MinTokens != nil {
		fmt.Fprintf(file, "    Min Tokens: %v\n", *params.MinTokens)
	}
	if params.TopP != nil {
		fmt.Fprintf(file, "    Top P: %v\n", *params.TopP)
	}
	if params.TopK != nil {
		fmt.Fprintf(file, "    Top K: %v\n", *params.TopK)
	}
	if params.MinP != nil {
		fmt.Fprintf(file, "    Min P: %v\n", *params.MinP)
	}
	if params.RepetitionPenalty != nil {
		fmt.Fprintf(file, "    Repetition Penalty: %v\n", *params.RepetitionPenalty)
	}
	if params.PresencePenalty != nil {
		fmt.Fprintf(file, "    Presence Penalty: %v\n", *params.PresencePenalty)
	}
	if params.FrequencyPenalty != nil {
		fmt.Fprintf(file, "    Frequency Penalty: %v\n", *params.FrequencyPenalty)
	}
	if params.Seed != nil {
		fmt.Fprintf(file, "    Seed: %v\n", *params.Seed)
	}
	if params.N != nil {
		fmt.Fprintf(file, "    N: %v\n", *params.N)
	}
	if len(params.Stop) > 0 {
		fmt.Fprintf(file, "    Stop: %v\n", params.Stop)
	}
	if len(params.StopTokenIds) > 0 {
		fmt.Fprintf(file, "    Stop Token IDs: %v\n", params.StopTokenIds)
	}
	if len(params.BadWords) > 0 {
		fmt.Fprintf(file, "    Bad Words: %v\n", params.BadWords)
	}
	if params.EnableThinking != nil {
		fmt.Fprintf(file, "    Enable Thinking: %v\n", *params.EnableThinking)
	}

	// vLLM specific parameters
	if len(params.GuidedChoice) > 0 {
		fmt.Fprintf(file, "    Guided Choice: %v\n", params.GuidedChoice)
	}
	if params.GuidedRegex != nil {
		fmt.Fprintf(file, "    Guided Regex: %v\n", *params.GuidedRegex)
	}
	if params.GuidedJSON != nil {
		fmt.Fprintf(file, "    Guided JSON: %v\n", params.GuidedJSON)
	}
	if params.GuidedGrammar != nil {
		fmt.Fprintf(file, "    Guided Grammar: %v\n", *params.GuidedGrammar)
	}

	// llama.cpp specific parameters
	if params.Mirostat != nil {
		fmt.Fprintf(file, "    Mirostat: %v\n", *params.Mirostat)
	}
	if params.MirostatTau != nil {
		fmt.Fprintf(file, "    Mirostat Tau: %v\n", *params.MirostatTau)
	}
	if params.MirostatEta != nil {
		fmt.Fprintf(file, "    Mirostat Eta: %v\n", *params.MirostatEta)
	}
	if params.TfsZ != nil {
		fmt.Fprintf(file, "    TFS Z: %v\n", *params.TfsZ)
	}
	if params.TypicalP != nil {
		fmt.Fprintf(file, "    Typical P: %v\n", *params.TypicalP)
	}

	fmt.Fprintf(file, "\n")

	// Write processing configuration
	fmt.Fprintf(file, "PROCESSING CONFIGURATION:\n")
	fmt.Fprintf(file, "  Workers: %d\n", p.config.Processing.Workers)
	fmt.Fprintf(file, "  Batch Size: %d\n", p.config.Processing.BatchSize)
	fmt.Fprintf(file, "  Repeat: %d\n", p.config.Processing.Repeat)
	fmt.Fprintf(file, "  Rate Limit: %v\n", p.config.Processing.RateLimit)
	if p.config.Processing.FlashInferSafe != nil {
		fmt.Fprintf(file, "  FlashInfer Safe: %v\n", *p.config.Processing.FlashInferSafe)
	}
	fmt.Fprintf(file, "  Minimal Mode: %v\n", p.config.Processing.MinimalMode)
	fmt.Fprintf(file, "\n")

	// Write output configuration
	fmt.Fprintf(file, "OUTPUT CONFIGURATION:\n")
	fmt.Fprintf(file, "  Directory: %s\n", p.config.Output.Directory)
	fmt.Fprintf(file, "  Format: %s\n", p.config.Output.Format)
	fmt.Fprintf(file, "  Include Raw Response: %v\n", p.config.Output.IncludeRawResponse)
	fmt.Fprintf(file, "  Include Thinking: %v\n", p.config.Output.IncludeThinking)
	fmt.Fprintf(file, "  Stream Output: %v\n", p.config.Output.StreamOutput)
	if p.config.Output.StreamSaveEvery > 0 {
		fmt.Fprintf(file, "  Stream Save Every: %d\n", p.config.Output.StreamSaveEvery)
	}
	fmt.Fprintf(file, "\n")

	// Write server information if available
	if serverInfo := p.getServerInfo(); serverInfo != nil {
		fmt.Fprintf(file, "SERVER INFORMATION:\n")
		fmt.Fprintf(file, "  Server URL: %s\n", serverInfo.ServerURL)
		fmt.Fprintf(file, "  Server Type: %s\n", serverInfo.ServerType)
		fmt.Fprintf(file, "  Available: %v\n", serverInfo.Available)
		fmt.Fprintf(file, "  Timestamp: %v\n", serverInfo.Timestamp)

		if len(serverInfo.Config) > 0 {
			fmt.Fprintf(file, "  Configuration:\n")
			for key, value := range serverInfo.Config {
				fmt.Fprintf(file, "    %s: %v\n", key, value)
			}
		}

		if len(serverInfo.Models) > 0 {
			fmt.Fprintf(file, "  Models:\n")
			for key, value := range serverInfo.Models {
				fmt.Fprintf(file, "    %s: %v\n", key, value)
			}
		}

		if len(serverInfo.Features) > 0 {
			fmt.Fprintf(file, "  Features:\n")
			for key, value := range serverInfo.Features {
				fmt.Fprintf(file, "    %s: %v\n", key, value)
			}
		}
	} else {
		fmt.Fprintf(file, "SERVER INFORMATION:\n")
		fmt.Fprintf(file, "  Status: Unavailable or client not initialized\n")
	}

	fmt.Fprintf(file, "End of configuration and server information\n")

	logger.Info("Configuration and server info saved to: %s", filename)
	return nil
}
