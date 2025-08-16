"""
GPU usage reporter for generating performance reports
"""
import json
import csv
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from .gpu_monitor import GPUMonitor


class GPUReporter:
    """Generate GPU usage reports from monitoring data"""
    
    def __init__(self, reports_dir: str = "gpu_reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)
        self.gpu_monitor = GPUMonitor()
    
    def generate_single_audio_report(self, 
                                   audio_filename: str, 
                                   process_func, 
                                   *args, 
                                   **kwargs) -> Dict[str, Any]:
        """
        Generate GPU usage report for processing a single audio file
        
        Args:
            audio_filename: Name of the audio file
            process_func: Function to process the audio (e.g., diarization service method)
            *args, **kwargs: Arguments to pass to process_func
        
        Returns:
            Dictionary containing processing result and GPU usage statistics
        """
        # Start GPU monitoring
        start_time = self.gpu_monitor.start_monitoring(audio_filename, "processing")
        
        try:
            # Run the audio processing function
            result = process_func(*args, **kwargs)
            processing_success = True
            error_message = None
        except Exception as e:
            result = None
            processing_success = False
            error_message = str(e)
        
        # Stop monitoring and get stats
        gpu_stats = self.gpu_monitor.stop_monitoring(audio_filename, "processing", start_time)
        
        # Compile report data
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'audio_filename': audio_filename,
            'processing_success': processing_success,
            'error_message': error_message,
            'gpu_stats': gpu_stats,
            'processing_result': result if processing_success else None
        }
        
        return report_data
    
    def generate_multiple_audio_report(self, 
                                     audio_files: List[str],
                                     process_func,
                                     *args, 
                                     **kwargs) -> Dict[str, Any]:
        """
        Generate GPU usage report for processing multiple audio files
        
        Args:
            audio_files: List of audio file names
            process_func: Function to process each audio file
            *args, **kwargs: Arguments to pass to process_func
        
        Returns:
            Dictionary containing results for all files and aggregate statistics
        """
        overall_start = time.time()
        individual_reports = []
        all_gpu_stats = []
        successful_processes = 0
        
        for audio_file in audio_files:
            report = self.generate_single_audio_report(audio_file, process_func, *args, **kwargs)
            individual_reports.append(report)
            
            if report['processing_success']:
                successful_processes += 1
            
            if report['gpu_stats']:
                all_gpu_stats.append(report['gpu_stats'])
        
        # Calculate aggregate statistics
        aggregate_stats = self._calculate_aggregate_stats(all_gpu_stats)
        
        # Compile multiple audio report
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'total_files': len(audio_files),
            'successful_processes': successful_processes,
            'total_duration': time.time() - overall_start,
            'individual_reports': individual_reports,
            'aggregate_gpu_stats': aggregate_stats
        }
        
        return report_data
    
    def generate_baseline_report(self, duration: int = 30) -> Dict[str, Any]:
        """
        Generate baseline GPU usage report (no audio processing)
        
        Args:
            duration: Duration in seconds to monitor baseline usage
        
        Returns:
            Dictionary containing baseline GPU statistics
        """
        baseline_filename = f"baseline_{int(time.time())}"
        
        # Start monitoring
        start_time = self.gpu_monitor.start_monitoring(baseline_filename, "baseline")
        
        # Wait for specified duration
        time.sleep(duration)
        
        # Stop monitoring and get stats
        gpu_stats = self.gpu_monitor.stop_monitoring(baseline_filename, "baseline", start_time)
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'type': 'baseline',
            'duration': duration,
            'gpu_stats': gpu_stats
        }
        
        return report_data
    
    def save_report(self, 
                    report_data: Dict[str, Any], 
                    report_name: str,
                    formats: List[str] = None) -> Dict[str, str]:
        """
        Save report in multiple formats
        
        Args:
            report_data: Report data dictionary
            report_name: Base name for the report files
            formats: List of formats to save ('json', 'txt', 'csv')
        
        Returns:
            Dictionary mapping format to saved file path
        """
        if formats is None:
            formats = ['json', 'txt']
        
        saved_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for format_type in formats:
            if format_type == 'json':
                file_path = self.reports_dir / f"{report_name}_{timestamp}.json"
                self._save_json_report(report_data, file_path)
                saved_files['json'] = str(file_path)
            
            elif format_type == 'txt':
                file_path = self.reports_dir / f"{report_name}_{timestamp}.txt"
                self._save_text_report(report_data, file_path)
                saved_files['txt'] = str(file_path)
            
            elif format_type == 'csv':
                file_path = self.reports_dir / f"{report_name}_{timestamp}.csv"
                self._save_csv_report(report_data, file_path)
                saved_files['csv'] = str(file_path)
        
        return saved_files
    
    def _calculate_aggregate_stats(self, stats_list: List[Dict]) -> Optional[Dict[str, Any]]:
        """Calculate aggregate statistics from multiple GPU stats"""
        if not stats_list:
            return None
        
        # Combine all nvidia-smi metrics
        all_nvidia_min_memory = [s['nvidia_min_memory'] for s in stats_list if 'nvidia_min_memory' in s]
        all_nvidia_max_memory = [s['nvidia_max_memory'] for s in stats_list if 'nvidia_max_memory' in s]
        all_nvidia_avg_memory = [s['nvidia_avg_memory'] for s in stats_list if 'nvidia_avg_memory' in s]
        
        all_nvidia_min_gpu = [s['nvidia_min_gpu_util'] for s in stats_list if 'nvidia_min_gpu_util' in s]
        all_nvidia_max_gpu = [s['nvidia_max_gpu_util'] for s in stats_list if 'nvidia_max_gpu_util' in s]
        all_nvidia_avg_gpu = [s['nvidia_avg_gpu_util'] for s in stats_list if 'nvidia_avg_gpu_util' in s]
        
        all_durations = [s['duration'] for s in stats_list if 'duration' in s]
        all_sample_counts = [s['sample_count'] for s in stats_list if 'sample_count' in s]
        
        return {
            'total_processes': len(stats_list),
            'nvidia_overall_min_memory': min(all_nvidia_min_memory) if all_nvidia_min_memory else 0,
            'nvidia_overall_max_memory': max(all_nvidia_max_memory) if all_nvidia_max_memory else 0,
            'nvidia_avg_memory_across_processes': sum(all_nvidia_avg_memory) / len(all_nvidia_avg_memory) if all_nvidia_avg_memory else 0,
            'nvidia_overall_min_gpu_util': min(all_nvidia_min_gpu) if all_nvidia_min_gpu else 0,
            'nvidia_overall_max_gpu_util': max(all_nvidia_max_gpu) if all_nvidia_max_gpu else 0,
            'nvidia_avg_gpu_util_across_processes': sum(all_nvidia_avg_gpu) / len(all_nvidia_avg_gpu) if all_nvidia_avg_gpu else 0,
            'total_duration': sum(all_durations) if all_durations else 0,
            'total_samples': sum(all_sample_counts) if all_sample_counts else 0
        }
    
    def _save_json_report(self, report_data: Dict[str, Any], file_path: Path):
        """Save report as JSON"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)
    
    def _save_text_report(self, report_data: Dict[str, Any], file_path: Path):
        """Save report as human-readable text"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("GPU USAGE PERFORMANCE REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Generated: {report_data.get('timestamp', 'Unknown')}\n")
            
            # Handle different report types
            if 'type' in report_data and report_data['type'] == 'baseline':
                self._write_baseline_text_report(f, report_data)
            elif 'individual_reports' in report_data:
                self._write_multiple_audio_text_report(f, report_data)
            elif 'performance_comparison' in report_data:
                self._write_comparison_text_report(f, report_data)
            else:
                self._write_single_audio_text_report(f, report_data)
    
    def _write_baseline_text_report(self, f, report_data):
        """Write baseline report section"""
        f.write(f"Report Type: Baseline GPU Usage\n")
        f.write(f"Duration: {report_data.get('duration', 0)} seconds\n\n")
        
        gpu_stats = report_data.get('gpu_stats')
        if gpu_stats:
            self._write_gpu_stats_text(f, gpu_stats, "Baseline")
        else:
            f.write("No GPU statistics available\n")
    
    def _write_single_audio_text_report(self, f, report_data):
        """Write single audio report section"""
        f.write(f"Report Type: Single Audio Processing\n")
        f.write(f"Audio File: {report_data.get('audio_filename', 'Unknown')}\n")
        f.write(f"Processing Success: {report_data.get('processing_success', False)}\n")
        
        if not report_data.get('processing_success', False):
            f.write(f"Error: {report_data.get('error_message', 'Unknown error')}\n")
        
        f.write("\n")
        
        gpu_stats = report_data.get('gpu_stats')
        if gpu_stats:
            self._write_gpu_stats_text(f, gpu_stats, "Processing")
        else:
            f.write("No GPU statistics available\n")
    
    def _write_multiple_audio_text_report(self, f, report_data):
        """Write multiple audio report section"""
        f.write(f"Report Type: Multiple Audio Processing\n")
        f.write(f"Total Files: {report_data.get('total_files', 0)}\n")
        f.write(f"Successful Processes: {report_data.get('successful_processes', 0)}\n")
        f.write(f"Total Duration: {report_data.get('total_duration', 0):.2f} seconds\n\n")
        
        # Aggregate statistics
        agg_stats = report_data.get('aggregate_gpu_stats')
        if agg_stats:
            f.write("AGGREGATE STATISTICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Processes: {agg_stats.get('total_processes', 0)}\n")
            f.write(f"Total Duration: {agg_stats.get('total_duration', 0):.2f} seconds\n")
            f.write(f"Total Samples: {agg_stats.get('total_samples', 0)}\n\n")
            
            f.write("GPU USAGE (nvidia-smi):\n")
            f.write(f"  Overall Memory Range: {agg_stats.get('nvidia_overall_min_memory', 0):.2f} - {agg_stats.get('nvidia_overall_max_memory', 0):.2f} GB\n")
            f.write(f"  Avg Memory Across Processes: {agg_stats.get('nvidia_avg_memory_across_processes', 0):.2f} GB\n")
            f.write(f"  Overall GPU Util Range: {agg_stats.get('nvidia_overall_min_gpu_util', 0):.1f} - {agg_stats.get('nvidia_overall_max_gpu_util', 0):.1f}%\n")
            f.write(f"  Avg GPU Util Across Processes: {agg_stats.get('nvidia_avg_gpu_util_across_processes', 0):.1f}%\n\n")
        
        # Individual results summary
        f.write("INDIVIDUAL RESULTS SUMMARY:\n")
        f.write("-" * 30 + "\n")
        individual_reports = report_data.get('individual_reports', [])
        for i, individual in enumerate(individual_reports, 1):
            f.write(f"{i}. {individual.get('audio_filename', 'Unknown')}: ")
            f.write(f"{'SUCCESS' if individual.get('processing_success') else 'FAILED'}\n")
            
            if individual.get('gpu_stats'):
                stats = individual['gpu_stats']
                f.write(f"   GPU Memory: {stats.get('nvidia_avg_memory', 0):.2f} GB avg, ")
                f.write(f"GPU Util: {stats.get('nvidia_avg_gpu_util', 0):.1f}% avg, ")
                f.write(f"Duration: {stats.get('duration', 0):.1f}s\n")
            else:
                f.write("   No GPU stats available\n")
        f.write("\n")
    
    def _write_comparison_text_report(self, f, report_data):
        """Write parallel vs sequential comparison report"""
        f.write(f"Report Type: Parallel vs Sequential Processing Comparison\n")
        f.write(f"Audio File: {report_data.get('audio_filename', 'Unknown')}\n\n")
        
        comparison = report_data.get('performance_comparison', {})
        
        f.write("PERFORMANCE COMPARISON:\n")
        f.write("=" * 30 + "\n")
        f.write(f"Parallel Processing Duration: {comparison.get('parallel_duration', 0):.2f} seconds\n")
        f.write(f"Sequential Processing Duration: {comparison.get('sequential_duration', 0):.2f} seconds\n")
        f.write(f"Speedup Factor: {comparison.get('speedup_factor', 0):.2f}x\n")
        f.write(f"Time Saved: {comparison.get('time_saved_seconds', 0):.2f} seconds\n\n")
        
        # Parallel processing details
        parallel_data = report_data.get('parallel_processing', {})
        if parallel_data and parallel_data.get('gpu_stats'):
            f.write("PARALLEL PROCESSING DETAILS:\n")
            f.write("-" * 30 + "\n")
            self._write_gpu_stats_text(f, parallel_data['gpu_stats'], "Parallel")
        
        # Sequential processing details  
        sequential_data = report_data.get('sequential_processing', {})
        if sequential_data and sequential_data.get('gpu_stats'):
            f.write("SEQUENTIAL PROCESSING DETAILS:\n")
            f.write("-" * 30 + "\n")
            self._write_gpu_stats_text(f, sequential_data['gpu_stats'], "Sequential")
    
    def _write_gpu_stats_text(self, f, gpu_stats, stage_name):
        """Write GPU statistics in text format"""
        f.write(f"GPU STATISTICS - {stage_name.upper()}:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Duration: {gpu_stats.get('duration', 0):.2f} seconds\n")
        f.write(f"Sample Count: {gpu_stats.get('sample_count', 0)}\n\n")
        
        f.write("GPU USAGE (nvidia-smi):\n")
        f.write(f"  Used Memory - Min: {gpu_stats.get('nvidia_min_memory', 0):.2f} GB, ")
        f.write(f"Max: {gpu_stats.get('nvidia_max_memory', 0):.2f} GB, ")
        f.write(f"Avg: {gpu_stats.get('nvidia_avg_memory', 0):.2f} GB\n")
        
        f.write(f"  GPU Compute Util - Min: {gpu_stats.get('nvidia_min_gpu_util', 0):.1f}%, ")
        f.write(f"Max: {gpu_stats.get('nvidia_max_gpu_util', 0):.1f}%, ")
        f.write(f"Avg: {gpu_stats.get('nvidia_avg_gpu_util', 0):.1f}%\n")
        
        f.write(f"  Memory Util - Min: {gpu_stats.get('nvidia_min_memory_util', 0):.1f}%, ")
        f.write(f"Max: {gpu_stats.get('nvidia_max_memory_util', 0):.1f}%, ")
        f.write(f"Avg: {gpu_stats.get('nvidia_avg_memory_util', 0):.1f}%\n\n")
    
    def _save_csv_report(self, report_data: Dict[str, Any], file_path: Path):
        """Save report as CSV (flattened data)"""
        # This is a simplified CSV format - can be expanded based on needs
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'timestamp', 'audio_filename', 'processing_success', 
                'nvidia_min_memory_gb', 'nvidia_max_memory_gb', 'nvidia_avg_memory_gb',
                'nvidia_min_gpu_util', 'nvidia_max_gpu_util', 'nvidia_avg_gpu_util',
                'nvidia_min_memory_util', 'nvidia_max_memory_util', 'nvidia_avg_memory_util',
                'duration_seconds', 'sample_count'
            ])
            
            # Handle different report types
            if 'individual_reports' in report_data:
                # Multiple audio report
                for individual in report_data.get('individual_reports', []):
                    self._write_csv_row(writer, individual)
            else:
                # Single audio or baseline report
                self._write_csv_row(writer, report_data)
    
    def _write_csv_row(self, writer, report_data):
        """Write a single CSV row"""
        gpu_stats = report_data.get('gpu_stats', {})
        
        row = [
            report_data.get('timestamp', ''),
            report_data.get('audio_filename', ''),
            report_data.get('processing_success', ''),
            gpu_stats.get('nvidia_min_memory', 0),
            gpu_stats.get('nvidia_max_memory', 0),
            gpu_stats.get('nvidia_avg_memory', 0),
            gpu_stats.get('nvidia_min_gpu_util', 0),
            gpu_stats.get('nvidia_max_gpu_util', 0),
            gpu_stats.get('nvidia_avg_gpu_util', 0),
            gpu_stats.get('nvidia_min_memory_util', 0),
            gpu_stats.get('nvidia_max_memory_util', 0),
            gpu_stats.get('nvidia_avg_memory_util', 0),
            gpu_stats.get('duration', 0),
            gpu_stats.get('sample_count', 0)
        ]
        
        writer.writerow(row)
