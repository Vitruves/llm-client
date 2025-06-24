package progress

import (
	"fmt"
	"llm-client/internal/logger"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"llm-client/internal/metrics"
)

type Progress struct {
	total        int64
	current      int64
	success      int64
	failed       int64
	startTime    time.Time
	lastUpdate   time.Time
	done         chan struct{}
	metricsCalc  *metrics.Calculator
	updateCount  int64
	ticker       *time.Ticker
	displayChan  chan struct{}
	lastPercent  float64
	mu           sync.Mutex
	isVisible    bool
	lastDisplay  string
	messageQueue chan string
}

func New(total int) *Progress {
	return NewWithMetrics(total, nil)
}

func NewWithMetrics(total int, metricsCalc *metrics.Calculator) *Progress {
	now := time.Now()
	return &Progress{
		total:        int64(total),
		startTime:    now,
		lastUpdate:   now,
		done:         make(chan struct{}),
		metricsCalc:  metricsCalc,
		displayChan:  make(chan struct{}, 1),
		lastPercent:  -1,
		messageQueue: make(chan string, 100),
	}
}

func (p *Progress) Start() {
	logger.Info("Starting process with %d items", p.total)
	p.setupStickyDisplay()
	p.ticker = time.NewTicker(100 * time.Millisecond)

	go func() {
		defer p.ticker.Stop()
		for {
			select {
			case <-p.done:
				p.displayFinal()
				return
			case <-p.ticker.C:
				select {
				case p.displayChan <- struct{}{}:
				default:
				}
			case msg := <-p.messageQueue:
				p.showMessage(msg)
			}
		}
	}()
}

// LogMessage adds a message to be displayed above the progress bar
func (p *Progress) LogMessage(message string) {
	select {
	case p.messageQueue <- message:
	default:
		// Queue is full, drop the message
	}
}

// setupStickyDisplay prepares the terminal for sticky bottom display
func (p *Progress) setupStickyDisplay() {
	// Save cursor position and hide cursor
	fmt.Print("\033[s\033[?25l")
	p.isVisible = true
	p.display()
}

// showMessage displays a message above the progress bar
func (p *Progress) showMessage(message string) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.isVisible {
		// Clear progress bar, show message, then redraw progress bar
		fmt.Print("\r\033[K") // Clear current line
		logger.Info(message)  // Print message
		p.redrawProgress()    // Redraw progress bar
	} else {
		logger.Info(message)
	}
}

// redrawProgress redraws the progress bar without going through the update logic
func (p *Progress) redrawProgress() {
	if p.lastDisplay != "" {
		fmt.Print(p.lastDisplay)
	}
}

func (p *Progress) Update(current, success, failed int) {
	atomic.StoreInt64(&p.current, int64(current))
	atomic.StoreInt64(&p.success, int64(success))
	atomic.StoreInt64(&p.failed, int64(failed))
	atomic.AddInt64(&p.updateCount, 1)
	p.triggerDisplay()
}

func (p *Progress) Increment() {
	atomic.AddInt64(&p.current, 1)
	atomic.AddInt64(&p.updateCount, 1)
	p.triggerDisplay()
}

func (p *Progress) triggerDisplay() {
	select {
	case p.displayChan <- struct{}{}:
	default:
	}
}

func (p *Progress) Stop() {
	p.mu.Lock()
	p.isVisible = false
	p.mu.Unlock()

	close(p.done)

	// Restore cursor and clear line
	fmt.Print("\r\033[K\033[?25h")
}

func (p *Progress) getTerminalWidth() int {
	// Try to get terminal width from environment
	if cols := os.Getenv("COLUMNS"); cols != "" {
		if w, err := strconv.Atoi(cols); err == nil && w > 0 {
			return w
		}
	}

	// Try using stty command
	if output, err := exec.Command("stty", "size").Output(); err == nil {
		if parts := strings.Fields(strings.TrimSpace(string(output))); len(parts) >= 2 {
			if w, err := strconv.Atoi(parts[1]); err == nil && w > 0 {
				return w
			}
		}
	}

	// Try using tput command
	if output, err := exec.Command("tput", "cols").Output(); err == nil {
		if w, err := strconv.Atoi(strings.TrimSpace(string(output))); err == nil && w > 0 {
			return w
		}
	}

	// Default fallback
	return 80
}

func (p *Progress) buildDots(current int64, availableWidth int) string {
	// Calculate maximum dots that fit in available width
	maxDots := availableWidth
	if maxDots < 10 {
		maxDots = 10 // Minimum dots
	}
	if maxDots > 60 {
		maxDots = 60 // Maximum dots for readability
	}

	dotsCount := int(float64(current) / float64(p.total) * float64(maxDots))
	if dotsCount > maxDots {
		dotsCount = maxDots
	}
	return strings.Repeat(".", dotsCount)
}

func formatDuration(d time.Duration) string {
	if d <= 0 {
		return "unknown"
	}

	seconds := int(d.Seconds())
	if seconds < 60 {
		return fmt.Sprintf("%ds", seconds)
	}

	minutes := seconds / 60
	if minutes < 60 {
		return fmt.Sprintf("%dm%ds", minutes, seconds%60)
	}

	hours := minutes / 60
	return fmt.Sprintf("%dh%dm", hours, minutes%60)
}

func formatThroughput(rate float64) string {
	if rate < 1 {
		return fmt.Sprintf("%.2f/s", rate)
	} else if rate < 100 {
		return fmt.Sprintf("%.1f/s", rate)
	} else {
		return fmt.Sprintf("%.0f/s", rate)
	}
}

func (p *Progress) getMetricText(current int64) string {
	if p.metricsCalc == nil || current == 0 {
		return ""
	}

	metric := p.metricsCalc.GetCurrentMetric()
	metricName := p.metricsCalc.GetMetricName()
	return fmt.Sprintf(" | %s%s%s: %s%.2f%s", logger.ColorPurple, metricName, logger.ColorReset, logger.ColorBold, metric, logger.ColorReset)
}

func (p *Progress) display() {
	p.mu.Lock()
	defer p.mu.Unlock()

	current := atomic.LoadInt64(&p.current)
	// success := atomic.LoadInt64(&p.success)
	failed := atomic.LoadInt64(&p.failed)

	if p.total == 0 {
		return // Avoid division by zero
	}

	percent := float64(current) / float64(p.total) * 100

	// Only redraw if percentage changed or 100ms passed since last display
	// This reduces flickering and CPU usage
	now := time.Now()
	if percent == p.lastPercent && now.Sub(p.lastUpdate) < 100*time.Millisecond && current != p.total {
		return
	}
	p.lastPercent = percent
	p.lastUpdate = now

	elapsed := time.Since(p.startTime)

	// Calculate estimated total time and remaining time
	var eta time.Duration
	var speed float64
	if current > 0 && elapsed > 0 {
		speed = float64(current) / elapsed.Seconds()
		// Check for division by zero before calculating eta
		if speed > 0 {
			eta = time.Duration((float64(p.total)/speed - elapsed.Seconds()) * float64(time.Second))
		}
	}

	durationStr := formatDuration(elapsed)
	etaStr := formatDuration(eta)
	speedStr := formatThroughput(speed)

	tbarWidth := p.getTerminalWidth() - 50 // Adjust based on prefix/suffix lengths
	if tbarWidth < 10 {
		tbarWidth = 10
	}

	dots := p.buildDots(current, tbarWidth)

	var barColor string
	var barChar = "─"

	if current == p.total {
		barColor = logger.ColorGreen // Green when complete
	} else if failed > 0 {
		barColor = logger.ColorRed // Red if there are errors
	} else if percent > 75 {
		barColor = logger.ColorYellow // Yellow for nearing completion
	} else {
		barColor = logger.ColorCyan // Cyan for in progress
	}

	progressBar := fmt.Sprintf("%s%s%s", barColor, strings.Repeat(barChar, len(dots)), logger.ColorReset)

	// stats := fmt.Sprintf(" %s%d%s/%s%d%s", logger.ColorGreen, success, logger.ColorReset, logger.ColorRed, failed, logger.ColorReset)

	progressLine := fmt.Sprintf("%s%.1f%%%s|%s%s| %s%d%s/%s%d%s [%s%s%s, %s%s] ETA: %s%s%s",
		logger.ColorBlue, percent, logger.ColorReset,
		progressBar, strings.Repeat(" ", tbarWidth-len(dots)),
		logger.ColorCyan, current, logger.ColorReset, logger.ColorBlue, p.total, logger.ColorReset,
		logger.ColorGray, durationStr, logger.ColorReset,
		logger.ColorGray, speedStr,
		logger.ColorGray, etaStr, logger.ColorReset)

	metricText := p.getMetricText(current)
	if metricText != "" {
		progressLine += metricText
	}

	p.lastDisplay = progressLine
	logger.Progress(progressLine)
}

func (p *Progress) displayFinal() {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Ensure progress bar is removed
	fmt.Print("\r\033[K\033[?25h")

	success := atomic.LoadInt64(&p.success)
	failed := atomic.LoadInt64(&p.failed)

	if failed > 0 {
		logger.Error("Completed with %d errors", failed)
	} else {
		logger.Success("Completed successfully with %d items", success)
	}
}
