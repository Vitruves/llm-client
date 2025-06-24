package progress

import (
	"fmt"
	"os"
	"os/signal"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/Vitruves/llm-client/internal/logger"

	"github.com/Vitruves/llm-client/internal/metrics"
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
	p.ticker = time.NewTicker(1 * time.Second)

	// Setup signal handler to restore cursor on interruption
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)

	go func() {
		defer p.ticker.Stop()
		defer p.restoreCursor() // Always restore cursor when goroutine exits

		for {
			select {
			case <-p.done:
				p.displayFinal()
				return
			case <-sigCh:
				// Handle interrupt signal - restore cursor and exit
				p.restoreCursor()
				return
			case <-p.ticker.C:
				select {
				case p.displayChan <- struct{}{}:
				default:
				}
			case <-p.displayChan:
				p.display()
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
	// Hide cursor for cleaner display
	fmt.Print("\033[?25l")
	p.isVisible = true
	p.display()
}

// showMessage displays a message above the progress bar
func (p *Progress) showMessage(message string) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.isVisible {
		// Clear current line completely
		fmt.Print("\r\033[K")
		// Print the message normally
		logger.Info(message)
		// Progress bar will be redrawn on next timer tick
	} else {
		logger.Info(message)
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
	// Only trigger if we're visible and not too frequently
	if !p.isVisible {
		return
	}
	select {
	case p.displayChan <- struct{}{}:
	default:
		// Channel full, skip this update
	}
}

// restoreCursor restores the terminal cursor
func (p *Progress) restoreCursor() {
	fmt.Print("\r\033[K\033[?25h")
}

func (p *Progress) Stop() {
	p.mu.Lock()
	p.isVisible = false
	p.mu.Unlock()

	close(p.done)

	// Restore cursor and clear line
	p.restoreCursor()
}

func formatDuration(d time.Duration) string {
	if d <= 0 {
		return "00:00"
	}

	seconds := int(d.Seconds())
	if seconds < 60 {
		return fmt.Sprintf("00:%02d", seconds)
	}

	minutes := seconds / 60
	if minutes < 60 {
		return fmt.Sprintf("%02d:%02d", minutes, seconds%60)
	}

	hours := minutes / 60
	minutes = minutes % 60
	return fmt.Sprintf("%02d:%02d:%02d", hours, minutes, seconds%3600%60)
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
	failed := atomic.LoadInt64(&p.failed)

	if p.total == 0 {
		return // Avoid division by zero
	}

	percent := float64(current) / float64(p.total) * 100

	// Only redraw if significant change or time passed
	now := time.Now()
	percentChanged := percent != p.lastPercent
	timeElapsed := now.Sub(p.lastUpdate) > 1*time.Second
	isComplete := current == p.total

	if !percentChanged && !timeElapsed && !isComplete {
		return
	}
	p.lastPercent = percent
	p.lastUpdate = now

	elapsed := time.Since(p.startTime)

	// Calculate ETA and speed
	var eta time.Duration
	var speed float64
	if current > 0 && elapsed > 0 {
		speed = float64(current) / elapsed.Seconds()
		if speed > 0 && current < p.total {
			eta = time.Duration((float64(p.total-current) / speed) * float64(time.Second))
		}
	}

	// Build progress line components
	timestamp := time.Now().Format("15:04")
	percentStr := fmt.Sprintf("%.1f%%", percent)
	durationStr := formatDuration(elapsed)
	etaStr := formatDuration(eta)
	speedStr := formatThroughput(speed)
	metricText := p.getMetricText(current)

	// Use a fixed bar width for consistency
	barWidth := 30 // Fixed width for better display

	// Build progress bar
	filledWidth := int(float64(barWidth) * percent / 100)
	if filledWidth > barWidth {
		filledWidth = barWidth
	}

	// Choose colors
	var barColor string
	if current == p.total {
		barColor = logger.ColorGreen
	} else if failed > 0 {
		barColor = logger.ColorRed
	} else if percent > 75 {
		barColor = logger.ColorYellow
	} else {
		barColor = logger.ColorCyan
	}

	// Build the progress bar with proper width
	progressBar := fmt.Sprintf("%s%s%s%s",
		barColor,
		strings.Repeat("█", filledWidth),
		logger.ColorReset,
		strings.Repeat("░", barWidth-filledWidth))

	// Build progress line with proper formatting and enhanced colors
	timestampColored := fmt.Sprintf("%s%s%s", logger.ColorBlue, timestamp, logger.ColorReset)
	infoColored := fmt.Sprintf("%s%sINFO%s%s", logger.ColorCyan, logger.ColorBold, logger.ColorReset, logger.ColorReset)
	percentColored := fmt.Sprintf("%s%s%s", logger.ColorYellow, percentStr, logger.ColorReset)
	countColored := fmt.Sprintf("%s%d%s/%s%d%s", logger.ColorGreen, current, logger.ColorReset, logger.ColorGreen, p.total, logger.ColorReset)
	timeColored := fmt.Sprintf("%s[%s, %s]%s", logger.ColorGray, durationStr, speedStr, logger.ColorReset)
	etaColored := fmt.Sprintf("ETA: %s%s%s", logger.ColorPurple, etaStr, logger.ColorReset)
	
	progressLine := fmt.Sprintf("%s - %s : %s|%s| %s %s %s%s",
		timestampColored,
		infoColored,
		percentColored,
		progressBar,
		countColored,
		timeColored,
		etaColored,
		metricText)

	// Don't truncate - let terminal handle wrapping
	// Just ensure we clear to end of line
	p.lastDisplay = progressLine

	// Clear to end of line and print progress
	fmt.Printf("\r%s\033[K", progressLine)
}

func (p *Progress) displayFinal() {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Clear progress bar completely and restore cursor
	p.restoreCursor()

	success := atomic.LoadInt64(&p.success)
	failed := atomic.LoadInt64(&p.failed)

	if failed > 0 {
		logger.Error("Completed with %d errors", failed)
	} else {
		logger.Success("Completed successfully with %d items", success)
	}
}
