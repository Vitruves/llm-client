package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/Vitruves/llm-client/internal/cli"
	"github.com/Vitruves/llm-client/internal/client"
	"github.com/Vitruves/llm-client/internal/config"
	"github.com/Vitruves/llm-client/internal/loader"
	"github.com/Vitruves/llm-client/internal/logger"
	"github.com/Vitruves/llm-client/internal/models"
	"github.com/Vitruves/llm-client/internal/parser"
	"github.com/Vitruves/llm-client/internal/processor"
	"github.com/Vitruves/llm-client/internal/reporter"

	"github.com/fatih/color"
	"github.com/spf13/cobra"
)

var version = "2.0.0"

func main() {
	if err := newRootCmd().Execute(); err != nil {
		cli.PrintError("%v", err)
		os.Exit(1)
	}
}

func newRootCmd() *cobra.Command {
	var rootCmd = &cobra.Command{
		Use:   "llm-client",
		Short: color.New(color.FgCyan, color.Bold).Sprint("A powerful, configurable client for LLM classification tasks"),
		Long: color.New(color.FgHiBlue, color.Bold).Sprint("LLM Client") + 
			 color.New(color.FgWhite).Sprint(" - A powerful, configurable client for LLM classification tasks\n\n") +
			 color.New(color.FgGreen, color.Bold).Sprint("Features:\n") +
			 color.New(color.FgYellow).Sprint("• Multiple LLM providers (vLLM, llama.cpp, OpenAI)\n") +
			 color.New(color.FgYellow).Sprint("• Concurrent processing with worker pools\n") +
			 color.New(color.FgYellow).Sprint("• Advanced parsing and consensus voting\n") +
			 color.New(color.FgYellow).Sprint("• Live metrics and progress tracking\n") +
			 color.New(color.FgYellow).Sprint("• Multiple output formats (JSON, CSV, Parquet, Excel)\n") +
			 color.New(color.FgYellow).Sprint("• Resumable processing with state management"),
		Version: version,
		PersistentPreRun: func(cmd *cobra.Command, args []string) {
			// Check for NO_COLOR environment variable
			if os.Getenv("NO_COLOR") != "" {
				color.NoColor = true
			}
		},
	}

	// Add global flags
	rootCmd.PersistentFlags().Bool("no-color", false, "Disable colored output")
	rootCmd.PersistentFlags().BoolP("help", "h", false, "Show help message")

	rootCmd.AddCommand(newRunCmd())
	rootCmd.AddCommand(newReportCmd())
	rootCmd.AddCommand(newHealthCmd())
	rootCmd.AddCommand(newConfigCmd())

	return rootCmd
}

func newRunCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "run",
		Short: color.New(color.FgGreen, color.Bold).Sprint("Run LLM processing on input data"),
		Long: color.New(color.FgHiBlue, color.Bold).Sprint("Process input data through Large Language Models\n\n") +
			 color.New(color.FgMagenta, color.Bold).Sprint("Supported Providers:\n") +
			 color.New(color.FgCyan).Sprint("• vLLM - High-performance inference server\n") +
			 color.New(color.FgCyan).Sprint("• llama.cpp - Efficient CPU/GPU inference\n") +
			 color.New(color.FgCyan).Sprint("• OpenAI - API-compatible endpoints\n\n") +
			 color.New(color.FgMagenta, color.Bold).Sprint("Examples:\n") +
			 color.New(color.FgYellow).Sprint("  llm-client run -c config.yaml -i data.csv\n") +
			 color.New(color.FgYellow).Sprint("  llm-client run -c config.yaml -i data.csv --temperature 0.7 --max-tokens 100 -w 50\n") +
			 color.New(color.FgYellow).Sprint("  llm-client run -c config.yaml -i data.csv --stream-output"),
		RunE: runClassify,
	}

	// Core flags
	cmd.Flags().StringP("config", "c", "config.yaml", "Configuration file path")
	cmd.Flags().StringP("input", "i", "", "Input data file (CSV/JSON/Excel/Parquet)")
	cmd.Flags().StringP("output", "o", "", "Output directory (overrides config)")
	cmd.Flags().IntP("workers", "w", 0, "Number of workers (overrides config)")
	cmd.Flags().IntP("repeat", "r", 0, "Repeat count for consensus (overrides config)")
	cmd.Flags().IntP("limit", "l", 0, "Limit processing to first N rows")
	cmd.Flags().Bool("progress", true, "Show progress bar")
	cmd.Flags().BoolP("verbose", "v", false, "Enable verbose output")
	cmd.Flags().Bool("dry-run", false, "Validate config without processing")
	cmd.Flags().String("resume", "", "Resume from saved state file")

	// Output control flags
	cmd.Flags().Bool("stream-output", false, "Enable streaming output (overrides config)")
	cmd.Flags().Int("stream-save-every", 0, "Force save every N results when streaming (overrides config)")
	cmd.Flags().Bool("minimal-mode", false, "Enable minimal processing mode (overrides config)")
	cmd.Flags().String("format", "", "Output format: json, csv, parquet, xlsx (overrides config)")

	// Model parameter overrides
	cmd.Flags().Float64("temperature", -1, "Temperature for sampling (0.0-2.0, overrides config)")
	cmd.Flags().Int("max-tokens", -1, "Maximum tokens to generate (overrides config)")
	cmd.Flags().Int("min-tokens", -1, "Minimum tokens to generate (overrides config)")
	cmd.Flags().Float64("top-p", -1, "Top-p nucleus sampling (0.0-1.0, overrides config)")
	cmd.Flags().Int("top-k", -1, "Top-k sampling (overrides config)")
	cmd.Flags().Float64("min-p", -1, "Min-p sampling (overrides config)")
	cmd.Flags().Float64("repetition-penalty", -1, "Repetition penalty (overrides config)")
	cmd.Flags().Float64("presence-penalty", -1, "Presence penalty (-2.0 to 2.0, overrides config)")
	cmd.Flags().Float64("frequency-penalty", -1, "Frequency penalty (-2.0 to 2.0, overrides config)")
	cmd.Flags().Int64("seed", -1, "Random seed for reproducibility (overrides config)")

	// Advanced sampling (llama.cpp)
	cmd.Flags().Int("mirostat", -1, "Mirostat sampling mode: 0=disabled, 1=Mirostat, 2=Mirostat 2.0")
	cmd.Flags().Float64("mirostat-tau", -1, "Mirostat target entropy (default: 5.0)")
	cmd.Flags().Float64("mirostat-eta", -1, "Mirostat learning rate (default: 0.1)")
	cmd.Flags().Float64("tfs-z", -1, "Tail-free sampling parameter (default: 1.0)")
	cmd.Flags().Float64("typical-p", -1, "Locally typical sampling probability (default: 1.0)")

	// vLLM specific features
	cmd.Flags().Bool("disable-thinking", false, "Disable thinking mode for Qwen3 (vLLM only, overrides config)")
	cmd.Flags().StringSlice("guided-choice", nil, "Constrain output to specific choices (vLLM)")
	cmd.Flags().String("guided-regex", "", "Enforce regex pattern on output (vLLM)")
	cmd.Flags().String("guided-grammar", "", "Apply grammar constraints (vLLM)")
	cmd.Flags().Bool("use-beam-search", false, "Use beam search instead of sampling (vLLM)")
	cmd.Flags().Int("best-of", -1, "Generate best_of completions and return best (vLLM)")

	// Stop conditions
	cmd.Flags().StringSlice("stop", nil, "Stop generation at these strings (overrides config)")
	cmd.Flags().IntSlice("stop-token-ids", nil, "Stop generation at these token IDs (overrides config)")

	// Custom validation function
	cmd.PreRunE = func(cmd *cobra.Command, args []string) error {
		resume, _ := cmd.Flags().GetString("resume")
		input, _ := cmd.Flags().GetString("input")

		if resume == "" && input == "" {
			return fmt.Errorf("either --input or --resume must be specified")
		}
		return nil
	}

	return cmd
}

func newReportCmd() *cobra.Command {
	var reportCmd = &cobra.Command{
		Use:   "report",
		Short: color.New(color.FgMagenta, color.Bold).Sprint("Generate reports from classification results"),
		Long:  color.New(color.FgHiBlue, color.Bold).Sprint("Analyze and generate detailed reports from LLM classification result files"),
	}

	reportCmd.AddCommand(newAnalyzeCmd())
	reportCmd.AddCommand(newCompareCmd())

	return reportCmd
}

func newAnalyzeCmd() *cobra.Command {
	var analyzeCmd = &cobra.Command{
		Use:   "analyze [result-file]",
		Short: "Analyze classification results",
		Args:  cobra.ExactArgs(1),
		RunE:  runAnalyze,
	}

	analyzeCmd.Flags().StringP("output", "o", "", "Output file path")
	analyzeCmd.Flags().String("format", "text", "Output format (text/json)")

	return analyzeCmd
}

func newCompareCmd() *cobra.Command {
	var compareCmd = &cobra.Command{
		Use:   "compare [file1] [file2]",
		Short: "Compare two result files",
		Args:  cobra.ExactArgs(2),
		RunE:  runCompare,
	}

	compareCmd.Flags().StringP("output", "o", "", "Output file path")
	compareCmd.Flags().String("format", "text", "Output format (text/json)")

	return compareCmd
}

func newHealthCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "health",
		Short: color.New(color.FgRed, color.Bold).Sprint("Check LLM server health and configuration"),
		Long: color.New(color.FgHiBlue, color.Bold).Sprint("Perform health checks on LLM servers\n\n") +
			 color.New(color.FgMagenta, color.Bold).Sprint("Examples:\n") +
			 color.New(color.FgYellow).Sprint("  llm-client health --vllm\n") +
			 color.New(color.FgYellow).Sprint("  llm-client health --llamacpp --get-server-config\n") +
			 color.New(color.FgYellow).Sprint("  llm-client health --curl-test http://localhost:8000"),
		RunE: runHealth,
	}

	// Health check flags
	cmd.Flags().Bool("vllm", false, "Check vLLM server health")
	cmd.Flags().Bool("llamacpp", false, "Check llama.cpp server health")
	cmd.Flags().Bool("get-server-config", false, "Retrieve and display server configuration")
	cmd.Flags().String("curl-test", "", "Test server endpoint with curl-like request")
	cmd.Flags().StringP("config", "c", "config.yaml", "Configuration file for server connection")
	cmd.Flags().BoolP("verbose", "v", false, "Enable verbose output")

	return cmd
}

func newConfigCmd() *cobra.Command {
	var configCmd = &cobra.Command{
		Use:   "config",
		Short: "Configuration utilities",
		Long:  "Validate configurations and test request/response processing",
	}

	configCmd.AddCommand(newConfigValidateCmd())

	return configCmd
}

func newConfigValidateCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "validate [config-file]",
		Short: "Validate configuration file",
		Long: `Validate configuration file and optionally test with sample data.

Examples:
  llm-client config validate config.yaml
  llm-client config validate config.yaml --test-file data.csv
  llm-client config validate config.yaml --test-file data.csv --show-request`,
		Args: cobra.ExactArgs(1),
		RunE: runConfigValidate,
	}

	// Validation flags
	cmd.Flags().String("test-file", "", "Test configuration with sample data file")
	cmd.Flags().Bool("show-request", false, "Display how data will be sent to server")
	cmd.Flags().Bool("show-response", false, "Display expected server response and parsing")
	cmd.Flags().Int("test-rows", 3, "Number of rows to test (default 3)")

	return cmd
}

func runClassify(cmd *cobra.Command, args []string) error {
	configFile, _ := cmd.Flags().GetString("config")
	inputFile, _ := cmd.Flags().GetString("input")
	outputDir, _ := cmd.Flags().GetString("output")
	limit, _ := cmd.Flags().GetInt("limit")
	workers, _ := cmd.Flags().GetInt("workers")
	repeat, _ := cmd.Flags().GetInt("repeat")
	verbose, _ := cmd.Flags().GetBool("verbose")
	showProgress, _ := cmd.Flags().GetBool("progress")
	dryRun, _ := cmd.Flags().GetBool("dry-run")
	resume, _ := cmd.Flags().GetString("resume")
	streamOutput, _ := cmd.Flags().GetBool("stream-output")
	streamSaveEvery, _ := cmd.Flags().GetInt("stream-save-every")
	minimalMode, _ := cmd.Flags().GetBool("minimal-mode")
	disableThinking, _ := cmd.Flags().GetBool("disable-thinking")

	// Only check input file existence if not resuming
	if resume == "" {
		if _, err := os.Stat(inputFile); os.IsNotExist(err) {
			return fmt.Errorf("input file does not exist: %s", inputFile)
		}
	}

	cfg, err := config.Load(configFile)
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}

	if outputDir != "" {
		cfg.Output.Directory = outputDir
	}
	if workers > 0 {
		cfg.Processing.Workers = workers
	}
	if repeat > 0 {
		cfg.Processing.Repeat = repeat
	}

	// Apply CLI flag overrides
	if cmd.Flags().Changed("stream-output") {
		cfg.Output.StreamOutput = streamOutput
	}
	if cmd.Flags().Changed("stream-save-every") {
		cfg.Output.StreamSaveEvery = streamSaveEvery
	}
	if cmd.Flags().Changed("minimal-mode") {
		cfg.Processing.MinimalMode = minimalMode
	}
	if cmd.Flags().Changed("disable-thinking") {
		// Only apply for vLLM provider
		if cfg.Provider.Name == "vllm" {
			if cfg.Model.Parameters.EnableThinking == nil {
				cfg.Model.Parameters.EnableThinking = new(bool)
			}
			*cfg.Model.Parameters.EnableThinking = !disableThinking // Set to false when disabling
		} else {
			cli.PrintWarning("disable-thinking flag only supported for vLLM provider, ignoring")
		}
	}

	// Apply output format override
	if format, _ := cmd.Flags().GetString("format"); format != "" {
		cfg.Output.Format = format
	}

	// Apply model parameter overrides
	applyModelParameterOverrides(cmd, &cfg.Model.Parameters, cfg.Provider.Name)

	if dryRun {
		cli.PrintSuccess("Configuration validation passed")
		printConfigSummary(cfg, configFile)
		return nil
	}

	proc := processor.New(cfg)
	proc.SetConfigFile(configFile)

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	opts := processor.Options{
		Limit:        limit,
		ShowProgress: showProgress,
		Verbose:      verbose,
		ResumeFile:   resume,
	}

	if resume != "" {
		cli.PrintInfo("Resuming from: %s", resume)
	} else {
		cli.PrintInfo("Starting classification: %s", inputFile)
	}

	if verbose {
		logger.SetVerbose(true)
		printConfigSummary(cfg, configFile)
	} else {
		logger.Info("Workers: %d, Repeat: %d, Provider: %s",
			cfg.Processing.Workers, cfg.Processing.Repeat, cfg.Provider.Name)
	}

	if err := proc.ProcessFile(ctx, inputFile, opts); err != nil {
		if strings.Contains(err.Error(), "processing cancelled") {
			// If processing was explicitly cancelled, we don't want to show the usage
			cmd.SilenceUsage = true
		}
		return fmt.Errorf("processing failed: %w", err)
	}

	cli.PrintSuccess("Processing completed successfully")
	return nil
}

func runAnalyze(cmd *cobra.Command, args []string) error {
	inputFile := args[0]
	output, _ := cmd.Flags().GetString("output")
	format, _ := cmd.Flags().GetString("format")

	results, err := loadResults(inputFile)
	if err != nil {
		return fmt.Errorf("failed to load results: %w", err)
	}

	rep := reporter.New(results)

	var content string
	switch format {
	case "json":
		content, err = rep.GenerateJSON()
	case "text":
		content = rep.GenerateText()
	default:
		return fmt.Errorf("unsupported format: %s", format)
	}

	if err != nil {
		return err
	}

	if output != "" {
		return os.WriteFile(output, []byte(content), 0644)
	}

	fmt.Print(content)
	return nil
}

func runCompare(cmd *cobra.Command, args []string) error {
	file1, file2 := args[0], args[1]
	output, _ := cmd.Flags().GetString("output")
	format, _ := cmd.Flags().GetString("format")

	results1, err := loadResults(file1)
	if err != nil {
		return fmt.Errorf("failed to load first file: %w", err)
	}

	results2, err := loadResults(file2)
	if err != nil {
		return fmt.Errorf("failed to load second file: %w", err)
	}

	comparison := generateComparison(results1, results2)

	var content string
	switch format {
	case "json":
		data, err := json.MarshalIndent(comparison, "", "  ")
		if err != nil {
			return err
		}
		content = string(data)
	case "text":
		content = formatComparison(comparison)
	default:
		return fmt.Errorf("unsupported format: %s", format)
	}

	if output != "" {
		return os.WriteFile(output, []byte(content), 0644)
	}

	fmt.Print(content)
	return nil
}

func loadResults(filename string) ([]models.Result, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	var output struct {
		Results []models.Result `json:"results"`
	}

	if err := json.Unmarshal(data, &output); err != nil {
		return nil, err
	}

	return output.Results, nil
}

type Comparison struct {
	File1 ComparisonStats `json:"file1"`
	File2 ComparisonStats `json:"file2"`
	Diff  DifferenceStats `json:"difference"`
}

type ComparisonStats struct {
	Total       int           `json:"total"`
	Success     int           `json:"success"`
	Failed      int           `json:"failed"`
	SuccessRate float64       `json:"success_rate"`
	AvgTime     time.Duration `json:"avg_time"`
}

type DifferenceStats struct {
	TotalDiff       int     `json:"total_diff"`
	SuccessRateDiff float64 `json:"success_rate_diff"`
	AvgTimeDiff     string  `json:"avg_time_diff"`
}

func generateComparison(results1, results2 []models.Result) *Comparison {
	stats1 := calculateComparisonStats(results1)
	stats2 := calculateComparisonStats(results2)

	diff := DifferenceStats{
		TotalDiff:       stats2.Total - stats1.Total,
		SuccessRateDiff: stats2.SuccessRate - stats1.SuccessRate,
		AvgTimeDiff:     (stats2.AvgTime - stats1.AvgTime).String(),
	}

	return &Comparison{
		File1: stats1,
		File2: stats2,
		Diff:  diff,
	}
}

func calculateComparisonStats(results []models.Result) ComparisonStats {
	if len(results) == 0 {
		return ComparisonStats{}
	}

	var successCount int
	var totalTime time.Duration

	for _, result := range results {
		if result.Success {
			successCount++
		}
		totalTime += result.ResponseTime
	}

	return ComparisonStats{
		Total:       len(results),
		Success:     successCount,
		Failed:      len(results) - successCount,
		SuccessRate: float64(successCount) / float64(len(results)) * 100,
		AvgTime:     totalTime / time.Duration(len(results)),
	}
}

func formatComparison(comp *Comparison) string {
	var report strings.Builder

	report.WriteString("COMPARISON REPORT\n")
	report.WriteString("=================\n\n")

	report.WriteString("FILE 1 STATS:\n")
	report.WriteString(fmt.Sprintf("  Total: %d\n", comp.File1.Total))
	report.WriteString(fmt.Sprintf("  Success: %d (%.2f%%)\n", comp.File1.Success, comp.File1.SuccessRate))
	report.WriteString(fmt.Sprintf("  Failed: %d\n", comp.File1.Failed))
	report.WriteString(fmt.Sprintf("  Avg Time: %v\n", comp.File1.AvgTime))

	report.WriteString("\nFILE 2 STATS:\n")
	report.WriteString(fmt.Sprintf("  Total: %d\n", comp.File2.Total))
	report.WriteString(fmt.Sprintf("  Success: %d (%.2f%%)\n", comp.File2.Success, comp.File2.SuccessRate))
	report.WriteString(fmt.Sprintf("  Failed: %d\n", comp.File2.Failed))
	report.WriteString(fmt.Sprintf("  Avg Time: %v\n", comp.File2.AvgTime))

	report.WriteString("\nDIFFERENCE:\n")
	report.WriteString(fmt.Sprintf("  Total Diff: %+d\n", comp.Diff.TotalDiff))
	report.WriteString(fmt.Sprintf("  Success Rate Diff: %+.2f%%\n", comp.Diff.SuccessRateDiff))
	report.WriteString(fmt.Sprintf("  Avg Time Diff: %s\n", comp.Diff.AvgTimeDiff))

	return report.String()
}

func applyModelParameterOverrides(cmd *cobra.Command, params *models.ModelParameters, providerName string) {
	// Basic sampling parameters
	if temp, _ := cmd.Flags().GetFloat64("temperature"); temp >= 0 {
		params.Temperature = &temp
	}
	if maxTokens, _ := cmd.Flags().GetInt("max-tokens"); maxTokens >= 0 {
		params.MaxTokens = &maxTokens
	}
	if minTokens, _ := cmd.Flags().GetInt("min-tokens"); minTokens >= 0 {
		params.MinTokens = &minTokens
	}
	if topP, _ := cmd.Flags().GetFloat64("top-p"); topP >= 0 {
		params.TopP = &topP
	}
	if topK, _ := cmd.Flags().GetInt("top-k"); topK >= 0 {
		params.TopK = &topK
	}
	if minP, _ := cmd.Flags().GetFloat64("min-p"); minP >= 0 {
		params.MinP = &minP
	}
	if repPenalty, _ := cmd.Flags().GetFloat64("repetition-penalty"); repPenalty >= 0 {
		params.RepetitionPenalty = &repPenalty
	}
	if presPenalty, _ := cmd.Flags().GetFloat64("presence-penalty"); presPenalty >= -2.0 {
		params.PresencePenalty = &presPenalty
	}
	if freqPenalty, _ := cmd.Flags().GetFloat64("frequency-penalty"); freqPenalty >= -2.0 {
		params.FrequencyPenalty = &freqPenalty
	}
	if seed, _ := cmd.Flags().GetInt64("seed"); seed >= 0 {
		params.Seed = &seed
	}

	// Stop conditions
	if stop, _ := cmd.Flags().GetStringSlice("stop"); len(stop) > 0 {
		params.Stop = stop
	}
	if stopTokenIds, _ := cmd.Flags().GetIntSlice("stop-token-ids"); len(stopTokenIds) > 0 {
		params.StopTokenIds = stopTokenIds
	}

	// llama.cpp specific parameters
	if providerName == "llamacpp" {
		if mirostat, _ := cmd.Flags().GetInt("mirostat"); mirostat >= 0 {
			params.Mirostat = &mirostat
		}
		if mirostatTau, _ := cmd.Flags().GetFloat64("mirostat-tau"); mirostatTau >= 0 {
			params.MirostatTau = &mirostatTau
		}
		if mirostatEta, _ := cmd.Flags().GetFloat64("mirostat-eta"); mirostatEta >= 0 {
			params.MirostatEta = &mirostatEta
		}
		if tfsZ, _ := cmd.Flags().GetFloat64("tfs-z"); tfsZ >= 0 {
			params.TfsZ = &tfsZ
		}
		if typicalP, _ := cmd.Flags().GetFloat64("typical-p"); typicalP >= 0 {
			params.TypicalP = &typicalP
		}
	}

	// vLLM specific parameters
	if providerName == "vllm" {
		if guidedChoice, _ := cmd.Flags().GetStringSlice("guided-choice"); len(guidedChoice) > 0 {
			params.GuidedChoice = guidedChoice
		}
		if guidedRegex, _ := cmd.Flags().GetString("guided-regex"); guidedRegex != "" {
			params.GuidedRegex = &guidedRegex
		}
		if guidedGrammar, _ := cmd.Flags().GetString("guided-grammar"); guidedGrammar != "" {
			params.GuidedGrammar = &guidedGrammar
		}
		if cmd.Flags().Changed("use-beam-search") {
			useBeamSearch, _ := cmd.Flags().GetBool("use-beam-search")
			params.UseBeamSearch = &useBeamSearch
		}
		if bestOf, _ := cmd.Flags().GetInt("best-of"); bestOf >= 0 {
			params.BestOf = &bestOf
		}
	}
}

func runHealth(cmd *cobra.Command, args []string) error {
	vllm, _ := cmd.Flags().GetBool("vllm")
	llamacpp, _ := cmd.Flags().GetBool("llamacpp")
	getServerConfig, _ := cmd.Flags().GetBool("get-server-config")
	curlTest, _ := cmd.Flags().GetString("curl-test")
	configFile, _ := cmd.Flags().GetString("config")
	verbose, _ := cmd.Flags().GetBool("verbose")

	if verbose {
		logger.SetVerbose(true)
	}

	if curlTest != "" {
		return runCurlTest(curlTest)
	}

	if vllm {
		return runVLLMHealthCheck(configFile, getServerConfig, true)
	}

	if llamacpp {
		return runLlamaCppHealthCheck(configFile, getServerConfig, true)
	}

	// If no specific flags, try to load config file
	if !vllm && !llamacpp && curlTest == "" {
		cfg, err := config.Load(configFile)
		if err != nil {
			cmd.SilenceUsage = true
			return fmt.Errorf("failed to load config: %w", err)
		}

		switch cfg.Provider.Name {
		case "vllm":
			return runVLLMHealthCheck(configFile, getServerConfig, false)
		case "llamacpp":
			return runLlamaCppHealthCheck(configFile, getServerConfig, false)
		default:
			cmd.SilenceUsage = true
			return fmt.Errorf("unsupported provider: %s", cfg.Provider.Name)
		}
	}

	cmd.SilenceUsage = true
	return fmt.Errorf("specify --vllm, --llamacpp, or --curl-test")
}

func runConfigValidate(cmd *cobra.Command, args []string) error {
	configFile := args[0]
	testFile, _ := cmd.Flags().GetString("test-file")
	showRequest, _ := cmd.Flags().GetBool("show-request")
	showResponse, _ := cmd.Flags().GetBool("show-response")
	testRows, _ := cmd.Flags().GetInt("test-rows")

	logger.Header("Configuration Validation")
	logger.Info("Validating: %s", configFile)

	// Load and validate configuration
	cfg, err := config.Load(configFile)
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}

	logger.Info("✓ Configuration loaded successfully")
	printConfigSummary(cfg, configFile)

	if testFile != "" {
		return runConfigTest(cfg, testFile, showRequest, showResponse, testRows)
	}

	logger.Success("Configuration validation completed")
	return nil
}

func runVLLMHealthCheck(configFile string, getServerConfig bool, useDefault bool) error {
	var cfg *models.Config
	var err error
	
	if useDefault && configFile == "config.yaml" {
		// Create default vLLM config
		cfg = &models.Config{
			Provider: models.ProviderConfig{
				Name:    "vllm",
				BaseURL: "http://localhost:8000",
				Timeout: "10s",
			},
			Model: models.ModelConfig{
				Name: "default",
			},
		}
		logger.Header("vLLM Health Check")
		logger.Info("Server: %s (using default)", cfg.Provider.BaseURL)
	} else {
		cfg, err = config.Load(configFile)
		if err != nil {
			return fmt.Errorf("failed to load config: %w", err)
		}

		if cfg.Provider.Name != "vllm" {
			return fmt.Errorf("config provider is %s, not vllm", cfg.Provider.Name)
		}

		logger.Header("vLLM Health Check")
		logger.Info("Server: %s", cfg.Provider.BaseURL)
	}

	client, err := client.NewClient(cfg)
	if err != nil {
		return fmt.Errorf("failed to create client: %w", err)
	}
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := client.HealthCheck(ctx); err != nil {
		logger.Error("Health check failed: %v", err)
		return err
	}

	logger.Success("✓ vLLM server is healthy")

	if getServerConfig {
		logger.Info("Retrieving server configuration...")
		serverInfo, err := client.GetServerInfo(ctx)
		if err != nil {
			logger.Warning("Failed to get server info: %v", err)
		} else {
			printServerInfo(serverInfo)
		}
	}

	return nil
}

func runLlamaCppHealthCheck(configFile string, getServerConfig bool, useDefault bool) error {
	var cfg *models.Config
	var err error
	
	if useDefault && configFile == "config.yaml" {
		// Create default llama.cpp config
		cfg = &models.Config{
			Provider: models.ProviderConfig{
				Name:    "llamacpp",
				BaseURL: "http://localhost:8080",
				Timeout: "10s",
			},
			Model: models.ModelConfig{
				Name: "default",
			},
		}
		logger.Header("llama.cpp Health Check")
		logger.Info("Server: %s (using default)", cfg.Provider.BaseURL)
	} else {
		cfg, err = config.Load(configFile)
		if err != nil {
			return fmt.Errorf("failed to load config: %w", err)
		}

		if cfg.Provider.Name != "llamacpp" {
			return fmt.Errorf("config provider is %s, not llamacpp", cfg.Provider.Name)
		}

		logger.Header("llama.cpp Health Check")
		logger.Info("Server: %s", cfg.Provider.BaseURL)
	}

	client, err := client.NewClient(cfg)
	if err != nil {
		return fmt.Errorf("failed to create client: %w", err)
	}
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := client.HealthCheck(ctx); err != nil {
		logger.Error("Health check failed: %v", err)
		return err
	}

	logger.Success("✓ llama.cpp server is healthy")

	if getServerConfig {
		logger.Info("Retrieving server configuration...")
		serverInfo, err := client.GetServerInfo(ctx)
		if err != nil {
			logger.Warning("Failed to get server info: %v", err)
		} else {
			printServerInfo(serverInfo)
		}
	}

	return nil
}

func runCurlTest(url string) error {
	logger.Header("cURL-like Test")
	logger.Info("Testing endpoint: %s", url)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Simple HTTP GET request
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		logger.Error("Request failed: %v", err)
		return err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read response: %w", err)
	}

	logger.Info("Status: %d %s", resp.StatusCode, resp.Status)
	logger.Info("Content-Type: %s", resp.Header.Get("Content-Type"))
	logger.Info("Content-Length: %d bytes", len(body))

	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		logger.Success("✓ Endpoint is accessible")
	} else {
		logger.Warning("⚠ Endpoint returned status %d", resp.StatusCode)
	}

	// Show response body if it's JSON and not too large
	if strings.Contains(resp.Header.Get("Content-Type"), "json") && len(body) < 2048 {
		logger.Debug("Response body: %s", string(body))
	}

	return nil
}

func runConfigTest(cfg *models.Config, testFile string, showRequest, showResponse bool, testRows int) error {
	logger.Header("Configuration Test")
	logger.Info("Testing with file: %s", testFile)
	
	// Enable verbose mode if we're showing request/response details
	if showRequest || showResponse {
		logger.SetVerbose(true)
	}

	// Load test data
	data, err := loader.LoadData(testFile)
	if err != nil {
		return fmt.Errorf("failed to load test data: %w", err)
	}

	if len(data) == 0 {
		return fmt.Errorf("test file contains no data")
	}

	// Limit test rows
	if testRows > 0 && testRows < len(data) {
		data = data[:testRows]
	}

	logger.Info("Testing with %d rows", len(data))

	// Create processor and parser for testing
	// proc := processor.New(cfg)
	parser := parser.New(cfg.Classification.Parsing)

	// Test template rendering and request generation
	for i, row := range data {
		logger.Info("--- Test Row %d ---", i+1)
		logger.Info("Input: %s", row.Text)

		if showRequest {
			// Show how template would be rendered
			systemPrompt := cfg.Classification.Template.System
			userPrompt := cfg.Classification.Template.User

			// Apply placeholders (this is a simplified version of what happens internally)
			renderedUser := strings.ReplaceAll(userPrompt, "{text}", row.Text)
			renderedUser = strings.ReplaceAll(renderedUser, "{index}", fmt.Sprintf("%d", row.Index))

			// Add any additional data fields
			for key, value := range row.Data {
				placeholder := fmt.Sprintf("{%s}", strings.ToLower(key))
				renderedUser = strings.ReplaceAll(renderedUser, placeholder, fmt.Sprintf("%v", value))
			}

			logger.Debug("System prompt: %s", systemPrompt)
			logger.Debug("User prompt: %s", renderedUser)
		}

		if showResponse {
			// Show expected parsing behavior with sample responses
			sampleResponses := []string{
				"positive",
				"The sentiment is negative",
				"I think this is neutral in tone",
			}

			for _, response := range sampleResponses {
				parsed := parser.Parse(response)
				logger.Debug("Sample response: '%s' -> Parsed: '%s'", response, parsed)
			}
		}
	}

	logger.Success("Configuration test completed")
	return nil
}

func printServerInfo(info *models.ServerInfo) {
	logger.Info("Server URL: %s", info.ServerURL)
	logger.Info("Server Type: %s", info.ServerType)
	logger.Info("Available: %v", info.Available)

	// Show actual server models
	if len(info.Models) > 0 {
		logger.Header("Server Models")
		if modelName, ok := info.Models["model_name"]; ok {
			logger.Info("Model Name: %v", modelName)
		}
		if maxModelLen, ok := info.Models["max_model_len"]; ok {
			logger.Info("Max Model Length: %v", maxModelLen)
		}
		if created, ok := info.Models["created"]; ok {
			logger.Info("Created: %v", created)
		}
		if ownedBy, ok := info.Models["owned_by"]; ok {
			logger.Info("Owned By: %v", ownedBy)
		}
	}

	// Show version information
	if len(info.Features) > 0 {
		if versionInfo, ok := info.Features["version_info"]; ok {
			logger.Header("Server Version")
			if versionMap, ok := versionInfo.(map[string]interface{}); ok {
				for key, value := range versionMap {
					logger.Info("%s: %v", key, value)
				}
			}
		}
		
		logger.Header("Server Features")
		for key, value := range info.Features {
			if key != "version_info" {
				logger.Info("%s: %v", key, value)
			}
		}
	}

	// Show client configuration
	if len(info.Config) > 0 {
		logger.Header("Client Configuration")
		
		// Group important parameters
		if temperature, ok := info.Config["temperature"]; ok {
			logger.Info("Temperature: %v", temperature)
		}
		if maxTokens, ok := info.Config["max_tokens"]; ok {
			logger.Info("Max Tokens: %v", maxTokens)
		}
		if topP, ok := info.Config["top_p"]; ok {
			logger.Info("Top-P: %v", topP)
		}
		if topK, ok := info.Config["top_k"]; ok {
			logger.Info("Top-K: %v", topK)
		}
		
		// Show all other parameters
		logger.Debug("All Client Parameters:")
		for key, value := range info.Config {
			if key != "temperature" && key != "max_tokens" && key != "top_p" && key != "top_k" {
				logger.Debug("  %s: %v", key, value)
			}
		}
	}
}

func printConfigSummary(cfg *models.Config, configFile string) {
	logger.Header("Configuration Summary")
	logger.Info("Config File: %s", configFile)
	logger.Info("Provider: %s (%s)", cfg.Provider.Name, cfg.Provider.BaseURL)
	logger.Info("Model: %s", cfg.Model.Name)
	logger.Info("Workers: %d, Repeat: %d", cfg.Processing.Workers, cfg.Processing.Repeat)
	logger.Info("Output: %s format to %s", cfg.Output.Format, cfg.Output.Directory)

	params := ""
	if cfg.Model.Parameters.Temperature != nil {
		params += fmt.Sprintf("Temperature: %.2f", *cfg.Model.Parameters.Temperature)
	}
	if cfg.Model.Parameters.MaxTokens != nil {
		if params != "" {
			params += ", "
		}
		params += fmt.Sprintf("Max Tokens: %d", *cfg.Model.Parameters.MaxTokens)
	}
	if cfg.Model.Parameters.TopP != nil {
		if params != "" {
			params += ", "
		}
		params += fmt.Sprintf("Top-P: %.2f", *cfg.Model.Parameters.TopP)
	}
	if params != "" {
		logger.Info("Parameters: %s", params)
	}
}
