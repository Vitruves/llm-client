#!/usr/bin/env python3
import json
import argparse
import sys
import signal
import math
from datetime import datetime
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

try:
    from argparse_color_formatter import ColorHelpFormatter
    colored_help = True
except ImportError:
    ColorHelpFormatter = argparse.HelpFormatter
    colored_help = False

try:
    from tqdm import tqdm
    tqdm_available = True
except ImportError:
    tqdm_available = False

class GracefulExit:
    """Handle SIGINT gracefully"""
    exit_now = False
    
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
    
    def exit_gracefully(self, signum, frame):
        self.exit_now = True
        print("\nReceived interrupt signal. Gracefully exiting...")

def is_nan_value(value) -> bool:
    """Check if value represents NaN in various formats"""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        return value.lower() in ['nan', 'null', 'none', '', 'n/a']
    return False

def safe_str_convert(value) -> Optional[str]:
    """Safely convert value to string, handling NaN cases"""
    if is_nan_value(value):
        return None
    return str(value).strip()

def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    if not text:
        return ""
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = ' '.join(text.split())
    return text

def calculate_ml_metrics(valid_predictions: List[str], valid_ground_truth: List[str]) -> Dict[str, Any]:
    """Calculate comprehensive ML classification metrics"""
    if not valid_predictions or not valid_ground_truth:
        return {}
    
    if len(valid_predictions) != len(valid_ground_truth):
        raise ValueError("Prediction and ground truth lists must have same length")
    
    labels = sorted(set(valid_ground_truth) | set(valid_predictions))
    total_valid = len(valid_predictions)
    accuracy = sum(1 for i in range(total_valid) if valid_predictions[i] == valid_ground_truth[i]) / total_valid
    
    per_class_metrics = {}
    confusion_matrix = {}
    
    for label in labels:
        tp = sum(1 for i in range(total_valid) if valid_ground_truth[i] == label and valid_predictions[i] == label)
        fp = sum(1 for i in range(total_valid) if valid_ground_truth[i] != label and valid_predictions[i] == label)
        fn = sum(1 for i in range(total_valid) if valid_ground_truth[i] == label and valid_predictions[i] != label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = sum(1 for gt in valid_ground_truth if gt == label)
        
        per_class_metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support
        }
        
        confusion_matrix[label] = {}
        for pred_label in labels:
            confusion_matrix[label][pred_label] = sum(
                1 for i in range(total_valid) 
                if valid_ground_truth[i] == label and valid_predictions[i] == pred_label
            )
    
    # Macro averages
    macro_precision = sum(m['precision'] for m in per_class_metrics.values()) / len(per_class_metrics)
    macro_recall = sum(m['recall'] for m in per_class_metrics.values()) / len(per_class_metrics)
    macro_f1 = sum(m['f1_score'] for m in per_class_metrics.values()) / len(per_class_metrics)
    
    # Weighted averages
    total_support = sum(m['support'] for m in per_class_metrics.values())
    weighted_precision = sum(m['precision'] * m['support'] for m in per_class_metrics.values()) / total_support if total_support > 0 else 0
    weighted_recall = sum(m['recall'] * m['support'] for m in per_class_metrics.values()) / total_support if total_support > 0 else 0
    weighted_f1 = sum(m['f1_score'] * m['support'] for m in per_class_metrics.values()) / total_support if total_support > 0 else 0
    
    # Cohen's Kappa
    po = accuracy
    pe = sum((sum(1 for gt in valid_ground_truth if gt == label) / total_valid) * 
             (sum(1 for pred in valid_predictions if pred == label) / total_valid) 
             for label in labels)
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0
    
    # Balanced accuracy
    class_recalls = [m['recall'] for m in per_class_metrics.values()]
    balanced_accuracy = sum(class_recalls) / len(class_recalls) if class_recalls else 0
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'cohen_kappa': kappa,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'per_class': per_class_metrics,
        'confusion_matrix': confusion_matrix,
        'labels': labels
    }

def write_text_report(output_file: Path, analysis_data: Dict[str, Any], format_type: str) -> None:
    """Write comprehensive analysis report to text file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("ML MODEL ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source: {analysis_data.get('source_file', 'Unknown')}\n")
        f.write(f"Format: {format_type}\n")
        f.write(f"Server: {analysis_data.get('server_info', {}).get('server_type', 'Unknown')}\n")
        f.write(f"Model: {analysis_data.get('model_name', 'Unknown')}\n\n")
        
        if 'performance_metrics' in analysis_data:
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-" * 40 + "\n")
            overall = analysis_data['performance_metrics']['overall']
            f.write(f"Total Samples: {overall['total_samples']}\n")
            f.write(f"Valid Samples: {overall['valid_samples']}\n")
            f.write(f"Missing/Invalid Samples: {overall['missing_samples']}\n")
            f.write(f"Success Rate: {overall['success_rate']:.4f}\n")
            f.write(f"Accuracy: {overall['accuracy']:.4f}\n")
            f.write(f"Balanced Accuracy: {overall['balanced_accuracy']:.4f}\n")
            f.write(f"Cohen's Kappa: {overall['cohen_kappa']:.4f}\n")
            
            if 'live_metrics' in analysis_data:
                live = analysis_data['live_metrics']
                f.write(f"Live {live['metric_name']}: {live['metric_value']:.2f}\n")
            f.write("\n")
            
            macro = analysis_data['performance_metrics']['macro_averages']
            weighted = analysis_data['performance_metrics']['weighted_averages']
            f.write("MACRO AVERAGES\n")
            f.write(f"Precision: {macro['precision']:.4f}\n")
            f.write(f"Recall: {macro['recall']:.4f}\n")
            f.write(f"F1-Score: {macro['f1_score']:.4f}\n\n")
            
            f.write("WEIGHTED AVERAGES\n")
            f.write(f"Precision: {weighted['precision']:.4f}\n")
            f.write(f"Recall: {weighted['recall']:.4f}\n")
            f.write(f"F1-Score: {weighted['f1_score']:.4f}\n\n")
            
            f.write("PER-CLASS METRICS\n")
            f.write("-" * 40 + "\n")
            per_class = analysis_data['performance_metrics']['per_class']
            for label, metrics in sorted(per_class.items()):
                f.write(f"Class {label}:\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall: {metrics['recall']:.4f}\n")
                f.write(f"  F1-Score: {metrics['f1_score']:.4f}\n")
                f.write(f"  Support: {metrics['support']}\n\n")
        
        if 'error_analysis' in analysis_data:
            error_summary = analysis_data['error_analysis']['summary']
            f.write("ERROR ANALYSIS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Errors: {error_summary['total_errors']}\n")
            f.write(f"Error Rate: {error_summary['error_rate']:.4f}\n")
            f.write(f"Failed Predictions: {error_summary['failed_predictions']}\n")
            f.write(f"NaN Values: {error_summary['nan_values']}\n\n")
            
            error_patterns = analysis_data['error_analysis']['error_patterns']['by_type']
            if error_patterns:
                f.write("ERROR PATTERNS BY TYPE\n")
                f.write("-" * 20 + "\n")
                for error_type, count in sorted(error_patterns.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"{error_type}: {count} cases\n")
                f.write("\n")
            
            response_stats = analysis_data['error_analysis'].get('response_time_stats', {})
            if response_stats:
                f.write("RESPONSE TIME STATISTICS\n")
                f.write("-" * 25 + "\n")
                f.write(f"Mean: {response_stats['mean']:.1f}ms\n")
                f.write(f"Median: {response_stats['median']:.1f}ms\n")
                f.write(f"Std Dev: {response_stats['std']:.1f}ms\n")
                f.write(f"Min: {response_stats['min']:.1f}ms\n")
                f.write(f"Max: {response_stats['max']:.1f}ms\n\n")
        
        # Format-specific content
        if format_type == 'discrepancies' and 'discrepancies' in analysis_data:
            f.write("DETAILED DISCREPANCIES\n")
            f.write("=" * 80 + "\n\n")
            
            for i, disc in enumerate(analysis_data['discrepancies'], 1):
                f.write(f"ERROR #{i}\n")
                f.write("-" * 50 + "\n")
                f.write(f"Index: {disc['index']}\n")
                f.write(f"Predicted: {disc['predicted']}\n")
                f.write(f"Ground Truth: {disc['ground_truth']}\n")
                f.write(f"Error Type: {disc['error_type']}\n")
                f.write(f"Success: {disc['success']}\n")
                f.write(f"Response Time: {disc['response_time_ms']:.1f}ms\n")
                f.write(f"Text Length: {disc['text_length']} characters\n")
                if disc.get('thinking_length'):
                    f.write(f"Thinking Length: {disc['thinking_length']} characters\n")
                f.write("\n")
                
                f.write("TEXT:\n")
                f.write(clean_text(disc['text']) + "\n\n")
                
                if disc.get('thinking'):
                    f.write("MODEL THINKING:\n")
                    f.write(clean_text(disc['thinking']) + "\n\n")
                
                f.write("=" * 80 + "\n\n")
        
        elif format_type == 'by_type' and 'errors_by_type' in analysis_data:
            f.write("ERRORS BY TYPE\n")
            f.write("=" * 80 + "\n\n")
            
            for error_type, errors in sorted(analysis_data['errors_by_type'].items()):
                f.write(f"ERROR TYPE: {error_type} ({len(errors)} cases)\n")
                f.write("=" * 60 + "\n\n")
                
                for i, error in enumerate(errors, 1):
                    f.write(f"Case #{i}\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"Index: {error['index']}\n")
                    f.write(f"Predicted: {error['predicted']}\n")
                    f.write(f"Ground Truth: {error['ground_truth']}\n")
                    f.write(f"Success: {error['success']}\n")
                    f.write(f"Response Time: {error['response_time_ms']:.1f}ms\n\n")
                    
                    f.write("TEXT:\n")
                    f.write(clean_text(error['text']) + "\n\n")
                    
                    if error.get('thinking'):
                        f.write("MODEL THINKING:\n")
                        f.write(clean_text(error['thinking']) + "\n\n")
                    
                    f.write("-" * 60 + "\n\n")

def extract_response_times(results: List[Dict[str, Any]]) -> List[float]:
    """Extract response times in milliseconds from results"""
    times = []
    for result in results:
        response_time = result.get('response_time', 0)
        if response_time and not is_nan_value(response_time):
            # Convert nanoseconds to milliseconds
            time_ms = float(response_time) / 1_000_000
            times.append(time_ms)
    return times

def calculate_response_time_stats(times: List[float]) -> Dict[str, float]:
    """Calculate response time statistics"""
    if not times:
        return {}
    
    import statistics
    return {
        'mean': statistics.mean(times),
        'median': statistics.median(times),
        'std': statistics.stdev(times) if len(times) > 1 else 0,
        'min': min(times),
        'max': max(times)
    }

def analyze_results(input_file: Path, output_format: str = 'comprehensive', 
                   prediction_col: str = 'final_answer', ground_truth_col: str = 'ground_truth') -> Path:
    """Analyze ML model results and generate comprehensive reports"""
    
    graceful_exit = GracefulExit()
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading file {input_file}: {e}")
        sys.exit(1)
    
    # Handle both legacy format and new format with nested results
    if 'results' in data:
        results = data['results']
        server_info = data.get('server_info', {})
        summary_info = data.get('summary', {})
        config_info = data.get('config', {})
    else:
        results = data if isinstance(data, list) else []
        server_info = {}
        summary_info = {}
        config_info = {}
    
    if not results:
        print("No results found in file")
        sys.exit(1)
    
    print(f"Processing {len(results)} results...")
    
    # Initialize containers
    all_records = []
    valid_predictions = []
    valid_ground_truth = []
    correct_predictions = []
    discrepancies = []
    missing_data = []
    failed_predictions = []
    
    # Setup progress bar if available
    if tqdm_available:
        iterator = tqdm(enumerate(results), total=len(results), 
                       desc="Analyzing results", position=0, leave=True)
    else:
        iterator = enumerate(results)
    
    for i, result in iterator:
        if graceful_exit.exit_now:
            print("Analysis interrupted by user")
            sys.exit(1)
        
        # Extract predictions and ground truth with NaN handling
        pred = safe_str_convert(result.get(prediction_col))
        gt = safe_str_convert(result.get(ground_truth_col))
        success = result.get('success', True)
        
        # Handle response time conversion
        response_time_raw = result.get('response_time', 0)
        response_time_ms = float(response_time_raw) / 1_000_000 if response_time_raw else 0
        
        base_record = {
            'result_index': i,
            'original_index': result.get('index', i),
            'prediction': pred,
            'ground_truth': gt,
            'text': result.get('input_text', ''),
            'text_length': len(result.get('input_text', '')),
            'thinking_content': result.get('thinking_content', ''),
            'thinking_length': len(result.get('thinking_content', '')),
            'response_time_ns': response_time_raw,
            'response_time_ms': response_time_ms,
            'success': success,
            'has_thinking': bool(result.get('thinking_content', ''))
        }
        
        all_records.append(base_record)
        
        # Categorize results
        if not success:
            failed_predictions.append(base_record)
        elif pred is None or gt is None:
            missing_type = 'prediction' if pred is None else 'ground_truth'
            if pred is None and gt is None:
                missing_type = 'both'
            
            missing_record = {
                **base_record,
                'missing_type': missing_type,
                'issue_description': f"Missing {missing_type}"
            }
            missing_data.append(missing_record)
        elif pred == gt:
            correct_predictions.append(base_record)
            valid_predictions.append(pred)
            valid_ground_truth.append(gt)
        else:
            # Determine error direction for numeric classes
            error_direction = 'different_class'
            try:
                pred_int = int(pred)
                gt_int = int(gt)
                if pred_int > gt_int:
                    error_direction = 'overestimate'
                elif pred_int < gt_int:
                    error_direction = 'underestimate'
            except (ValueError, TypeError):
                pass
            
            discrepancy_record = {
                **base_record,
                'error_type': f"{gt}_to_{pred}",
                'error_direction': error_direction
            }
            discrepancies.append(discrepancy_record)
            valid_predictions.append(pred)
            valid_ground_truth.append(gt)
    
    # Calculate ML metrics
    ml_metrics = calculate_ml_metrics(valid_predictions, valid_ground_truth)
    
    # Calculate additional statistics
    error_patterns = Counter(d['error_type'] for d in discrepancies)
    error_directions = Counter(d['error_direction'] for d in discrepancies)
    response_times = extract_response_times(results)
    response_time_stats = calculate_response_time_stats(response_times)
    
    # Extract model information
    model_name = 'Unknown'
    if config_info:
        model_name = config_info.get('Model', {}).get('Name', 'Unknown')
    elif server_info:
        model_name = server_info.get('models', {}).get('model_name', 'Unknown')
    
    # Build analysis data structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_samples = len(results)
    valid_samples = len(valid_predictions)
    
    analysis_data = {
        'source_file': str(input_file),
        'model_name': model_name,
        'server_info': server_info,
        'performance_metrics': {
            'overall': {
                'accuracy': ml_metrics.get('accuracy', 0),
                'balanced_accuracy': ml_metrics.get('balanced_accuracy', 0),
                'cohen_kappa': ml_metrics.get('cohen_kappa', 0),
                'total_samples': total_samples,
                'valid_samples': valid_samples,
                'missing_samples': len(missing_data),
                'success_rate': (total_samples - len(failed_predictions)) / total_samples if total_samples > 0 else 0
            },
            'macro_averages': {
                'precision': ml_metrics.get('macro_precision', 0),
                'recall': ml_metrics.get('macro_recall', 0),
                'f1_score': ml_metrics.get('macro_f1', 0)
            },
            'weighted_averages': {
                'precision': ml_metrics.get('weighted_precision', 0),
                'recall': ml_metrics.get('weighted_recall', 0),
                'f1_score': ml_metrics.get('weighted_f1', 0)
            },
            'per_class': ml_metrics.get('per_class', {})
        },
        'error_analysis': {
            'summary': {
                'total_errors': len(discrepancies),
                'error_rate': len(discrepancies) / valid_samples if valid_samples > 0 else 0,
                'failed_predictions': len(failed_predictions),
                'nan_values': len(missing_data)
            },
            'error_patterns': {
                'by_type': dict(error_patterns),
                'by_direction': dict(error_directions)
            },
            'response_time_stats': response_time_stats
        }
    }
    
    # Add live metrics if available
    if summary_info and 'live_metrics' in summary_info:
        analysis_data['live_metrics'] = summary_info['live_metrics']
    
    # Generate output based on format
    output_file = generate_output_file(input_file, output_format, timestamp, analysis_data, discrepancies)
    
    write_text_report(output_file, analysis_data, output_format)
    
    # Print summary
    print_analysis_summary(output_file, analysis_data, error_patterns)
    
    return output_file

def generate_output_file(input_file: Path, output_format: str, timestamp: str, 
                        analysis_data: Dict[str, Any], discrepancies: List[Dict[str, Any]]) -> Path:
    """Generate output filename and prepare format-specific data"""
    
    if output_format == 'discrepancies':
        clean_discrepancies = []
        for d in discrepancies:
            clean_record = {
                'index': d['original_index'],
                'predicted': d['prediction'],
                'ground_truth': d['ground_truth'],
                'error_type': d['error_type'],
                'text': d['text'],
                'thinking': d['thinking_content'],
                'response_time_ms': d['response_time_ms'],
                'text_length': d['text_length'],
                'thinking_length': d['thinking_length'],
                'success': d['success']
            }
            clean_discrepancies.append(clean_record)
        analysis_data['discrepancies'] = clean_discrepancies
        output_file = input_file.with_name(f"{input_file.stem}_discrepancies_{timestamp}.txt")
        
    elif output_format == 'minimal':
        clean_discrepancies = []
        for d in discrepancies:
            clean_record = {
                'index': d['original_index'],
                'predicted': d['prediction'],
                'ground_truth': d['ground_truth'],
                'text': d['text'],
                'thinking': d['thinking_content']
            }
            clean_discrepancies.append(clean_record)
        analysis_data['discrepancies'] = clean_discrepancies
        output_file = input_file.with_name(f"{input_file.stem}_errors_minimal_{timestamp}.txt")
        
    elif output_format == 'by_type':
        errors_by_type = defaultdict(list)
        for d in discrepancies:
            clean_record = {
                'index': d['original_index'],
                'predicted': d['prediction'],
                'ground_truth': d['ground_truth'],
                'text': d['text'],
                'thinking': d['thinking_content'],
                'response_time_ms': d['response_time_ms'],
                'success': d['success']
            }
            errors_by_type[d['error_type']].append(clean_record)
        analysis_data['errors_by_type'] = dict(errors_by_type)
        output_file = input_file.with_name(f"{input_file.stem}_errors_by_type_{timestamp}.txt")
        
    else:  # comprehensive
        clean_discrepancies = []
        for d in discrepancies:
            clean_record = {
                'index': d['original_index'],
                'predicted': d['prediction'],
                'ground_truth': d['ground_truth'],
                'error_type': d['error_type'],
                'error_direction': d['error_direction'],
                'text': d['text'],
                'thinking': d['thinking_content'],
                'response_time_ms': d['response_time_ms'],
                'text_length': d['text_length'],
                'thinking_length': d['thinking_length'],
                'success': d['success']
            }
            clean_discrepancies.append(clean_record)
        
        analysis_data['detailed_records'] = {
            'discrepancies': clean_discrepancies
        }
        output_file = input_file.with_name(f"{input_file.stem}_analysis_{timestamp}.txt")
    
    return output_file

def print_analysis_summary(output_file: Path, analysis_data: Dict[str, Any], error_patterns: Counter) -> None:
    """Print analysis summary to console"""
    overall = analysis_data['performance_metrics']['overall']
    error_summary = analysis_data['error_analysis']['summary']
    
    print(f"\nAnalysis complete")
    print(f"Output file: {output_file}")
    print(f"Model: {analysis_data['model_name']}")
    print(f"Total samples: {overall['total_samples']}")
    print(f"Valid samples: {overall['valid_samples']}")
    print(f"Success rate: {overall['success_rate']:.4f}")
    print(f"Accuracy: {overall['accuracy']:.4f}")
    print(f"Balanced accuracy: {overall['balanced_accuracy']:.4f}")
    print(f"Cohen's Kappa: {overall['cohen_kappa']:.4f}")
    print(f"Errors: {error_summary['total_errors']}")
    print(f"Failed predictions: {error_summary['failed_predictions']}")
    print(f"Missing/NaN values: {error_summary['nan_values']}")
    
    if 'live_metrics' in analysis_data:
        live = analysis_data['live_metrics']
        print(f"Live {live['metric_name']}: {live['metric_value']:.2f}")
    
    if error_patterns:
        print("\nTop error patterns:")
        for error_type, count in error_patterns.most_common(5):
            print(f"  {error_type}: {count}")
    
    response_stats = analysis_data['error_analysis'].get('response_time_stats', {})
    if response_stats:
        print(f"\nResponse time: {response_stats['mean']:.1f}ms ± {response_stats['std']:.1f}ms")

def main():
    parser = argparse.ArgumentParser(
        description='Analyze ML model results and generate comprehensive human-readable reports',
        formatter_class=ColorHelpFormatter if colored_help else argparse.HelpFormatter
    )
    
    parser.add_argument(
        'input_file', 
        type=Path,
        help='Input JSON file with ML model results'
    )
    
    parser.add_argument(
        '-f', '--format', 
        choices=['comprehensive', 'discrepancies', 'minimal', 'by_type'], 
        default='discrepancies',
        help='Output report format (default: discrepancies)'
    )
    
    parser.add_argument(
        '-p', '--prediction-col', 
        default='final_answer',
        help='Column name containing model predictions (default: final_answer)'
    )
    
    parser.add_argument(
        '-g', '--ground-truth-col', 
        default='ground_truth',
        help='Column name containing ground truth labels (default: ground_truth)'
    )
    
    if not colored_help:
        print("Note: Install 'argparse-color-formatter' for colored help output")
    if not tqdm_available:
        print("Note: Install 'tqdm' for progress bars")
    
    args = parser.parse_args()
    
    if not args.input_file.exists():
        print(f"Error: Input file '{args.input_file}' does not exist")
        sys.exit(1)
    
    try:
        analyze_results(args.input_file, args.format, args.prediction_col, args.ground_truth_col)
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()