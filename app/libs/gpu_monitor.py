"""
GPU Monitoring System for Cog Predictor
Provides real-time GPU usage monitoring using nvidia-smi
"""

import os
import time
import threading
import queue
import subprocess


class GPUMonitor:
    """Real-time GPU monitoring system using nvidia-smi"""
    
    def __init__(self):
        self.monitoring_active = False
        self.monitor_thread = None
        self.monitor_queue = queue.Queue()
    
    def get_gpu_usage(self):
        """Get current GPU memory usage and utilization using nvidia-smi"""
        try:
            # Get real GPU usage using nvidia-smi
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                memory_used_mb = float(values[0])
                memory_total_mb = float(values[1])
                gpu_util = float(values[2])
                memory_util = float(values[3])
                
                return {
                    "nvidia_smi_used_memory_gb": memory_used_mb / 1024,
                    "nvidia_smi_total_memory_gb": memory_total_mb / 1024,
                    "nvidia_smi_memory_percent": (memory_used_mb / memory_total_mb) * 100,
                    "nvidia_smi_gpu_utilization": gpu_util,
                    "nvidia_smi_memory_utilization": memory_util
                }
            else:
                print(f"nvidia-smi failed with return code: {result.returncode}")
                return None
                
        except Exception as e:
            print(f"Error getting GPU usage: {e}")
            return None
    
    def start_monitoring(self, audio_filename, stage, interval=0.5):
        """Start concurrent GPU monitoring in a separate thread"""
        if self.monitoring_active:
            self.stop_monitoring(audio_filename, stage, time.time())
        
        self.monitoring_active = True
        self.monitor_queue = queue.Queue()
        start_time = time.time()
        
        def monitor_worker():
            sample_count = 0
            
            while self.monitoring_active:
                current_time = time.time()
                elapsed_time = current_time - start_time
                sample_count += 1
                
                gpu_usage = self.get_gpu_usage()
                
                # Put the data in queue for later processing
                self.monitor_queue.put({
                    'timestamp': current_time,
                    'elapsed_time': elapsed_time,
                    'sample_number': sample_count,
                    'gpu_usage': gpu_usage
                })
                
                time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor_worker, daemon=True)
        self.monitor_thread.start()
        
        return start_time

    def stop_monitoring(self, audio_filename, stage, start_time):
        """Stop GPU monitoring and save the collected data"""
        if not self.monitoring_active:
            return None
            
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        # Process collected data
        samples = []
        while not self.monitor_queue.empty():
            try:
                sample = self.monitor_queue.get_nowait()
                samples.append(sample)
            except queue.Empty:
                break
        
        if samples:
            stats = self._calculate_statistics(samples)
            self.save_monitoring_data(audio_filename, stage, samples, start_time)
            return stats
        
        return None

    def _calculate_statistics(self, samples):
        """Calculate statistics from monitoring samples"""
        gpu_usages = [s['gpu_usage'] for s in samples if s['gpu_usage']]
        if not gpu_usages:
            return None
            
        # nvidia-smi metrics only
        nvidia_memories = [g.get('nvidia_smi_used_memory_gb', 0) for g in gpu_usages if g.get('nvidia_smi_used_memory_gb') is not None]
        nvidia_gpu_utils = [g.get('nvidia_smi_gpu_utilization', 0) for g in gpu_usages if g.get('nvidia_smi_gpu_utilization') is not None]
        nvidia_memory_utils = [g.get('nvidia_smi_memory_utilization', 0) for g in gpu_usages if g.get('nvidia_smi_memory_utilization') is not None]
        
        stats = {
            # nvidia-smi stats only
            'nvidia_min_memory': min(nvidia_memories) if nvidia_memories else 0,
            'nvidia_max_memory': max(nvidia_memories) if nvidia_memories else 0,
            'nvidia_avg_memory': sum(nvidia_memories) / len(nvidia_memories) if nvidia_memories else 0,
            'nvidia_min_gpu_util': min(nvidia_gpu_utils) if nvidia_gpu_utils else 0,
            'nvidia_max_gpu_util': max(nvidia_gpu_utils) if nvidia_gpu_utils else 0,
            'nvidia_avg_gpu_util': sum(nvidia_gpu_utils) / len(nvidia_gpu_utils) if nvidia_gpu_utils else 0,
            'nvidia_min_memory_util': min(nvidia_memory_utils) if nvidia_memory_utils else 0,
            'nvidia_max_memory_util': max(nvidia_memory_utils) if nvidia_memory_utils else 0,
            'nvidia_avg_memory_util': sum(nvidia_memory_utils) / len(nvidia_memory_utils) if nvidia_memory_utils else 0,
            
            'sample_count': len(samples),
            'duration': samples[-1]['elapsed_time'] if samples else 0
        }
        
        return stats

    def save_monitoring_data(self, audio_filename, stage, samples, start_time):
        """Save concurrent GPU monitoring data to file"""
        if not samples:
            return
            
        # Create GPU monitor output directory if it doesn't exist
        os.makedirs("gpu_monitor_output", exist_ok=True)
        
        # Generate timestamp for the log entry
        timestamp = int(start_time)
        log_filename = f"gpu_monitor_output/{audio_filename}_gpu_usage_{timestamp}.txt"
        
        # Calculate statistics from nvidia-smi metrics only
        gpu_usages = [s['gpu_usage'] for s in samples if s['gpu_usage']]
        if gpu_usages:
            # nvidia-smi metrics only
            nvidia_memories = [g.get('nvidia_smi_used_memory_gb', 0) for g in gpu_usages if g.get('nvidia_smi_used_memory_gb') is not None]
            nvidia_gpu_utils = [g.get('nvidia_smi_gpu_utilization', 0) for g in gpu_usages if g.get('nvidia_smi_gpu_utilization') is not None]
            nvidia_memory_utils = [g.get('nvidia_smi_memory_utilization', 0) for g in gpu_usages if g.get('nvidia_smi_memory_utilization') is not None]
            
            stats = {
                # nvidia-smi stats only
                'nvidia_min_memory': min(nvidia_memories) if nvidia_memories else 0,
                'nvidia_max_memory': max(nvidia_memories) if nvidia_memories else 0,
                'nvidia_avg_memory': sum(nvidia_memories) / len(nvidia_memories) if nvidia_memories else 0,
                'nvidia_min_gpu_util': min(nvidia_gpu_utils) if nvidia_gpu_utils else 0,
                'nvidia_max_gpu_util': max(nvidia_gpu_utils) if nvidia_gpu_utils else 0,
                'nvidia_avg_gpu_util': sum(nvidia_gpu_utils) / len(nvidia_gpu_utils) if nvidia_gpu_utils else 0,
                'nvidia_min_memory_util': min(nvidia_memory_utils) if nvidia_memory_utils else 0,
                'nvidia_max_memory_util': max(nvidia_memory_utils) if nvidia_memory_utils else 0,
                'nvidia_avg_memory_util': sum(nvidia_memory_utils) / len(nvidia_memory_utils) if nvidia_memory_utils else 0,
                
                'sample_count': len(samples),
                'duration': samples[-1]['elapsed_time'] if samples else 0
            }
        
        with open(log_filename, "a", encoding="utf-8") as f:
            f.write(f"Stage: {stage} (CONCURRENT MONITORING)\n")
            f.write(f"  Monitoring Duration: {stats['duration']:.2f} seconds\n")
            f.write(f"  Sample Count: {stats['sample_count']}\n")
            
            if gpu_usages:
                total_memory = gpu_usages[0].get('nvidia_smi_total_memory_gb', 0)
                f.write(f"  Total GPU Memory: {total_memory:.2f} GB\n")
                
                f.write(f"\n  GPU USAGE (nvidia-smi):\n")
                f.write(f"    Used Memory - Min: {stats['nvidia_min_memory']:.2f} GB, Max: {stats['nvidia_max_memory']:.2f} GB, Avg: {stats['nvidia_avg_memory']:.2f} GB\n")
                f.write(f"    GPU Compute Utilization - Min: {stats['nvidia_min_gpu_util']:.1f}%, Max: {stats['nvidia_max_gpu_util']:.1f}%, Avg: {stats['nvidia_avg_gpu_util']:.1f}%\n")
                f.write(f"    Memory Utilization - Min: {stats['nvidia_min_memory_util']:.1f}%, Max: {stats['nvidia_max_memory_util']:.1f}%, Avg: {stats['nvidia_avg_memory_util']:.1f}%\n")
            
            f.write(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
            f.write("-" * 30 + "\n\n")
            
            # Write detailed samples (every 5th sample to avoid too much data)
            f.write(f"DETAILED SAMPLES FOR {stage}:\n")
            for i, sample in enumerate(samples[::5]):
                if sample['gpu_usage']:
                    gpu = sample['gpu_usage']
                    nvidia_mem = gpu.get('nvidia_smi_used_memory_gb', 'N/A')
                    nvidia_gpu_util = gpu.get('nvidia_smi_gpu_utilization', 'N/A')
                    nvidia_mem_util = gpu.get('nvidia_smi_memory_utilization', 'N/A')
                    f.write(f"  Sample {sample['sample_number']}: {sample['elapsed_time']:.1f}s - "
                           f"Memory: {nvidia_mem:.2f}GB, "
                           f"GPU Util: {nvidia_gpu_util}%, "
                           f"Mem Util: {nvidia_mem_util}%\n")
            f.write("-" * 50 + "\n\n")

    def log_snapshot(self, audio_filename, stage, additional_info=None):
        """Log a single GPU usage snapshot to a text file"""
        gpu_usage = self.get_gpu_usage()
        
        # Create GPU monitor output directory if it doesn't exist
        os.makedirs("gpu_monitor_output", exist_ok=True)
        
        # Generate timestamp for the log entry
        timestamp = int(time.time())
        log_filename = f"gpu_monitor_output/{audio_filename}_gpu_usage_{timestamp}.txt"
        
        # Check if file exists to determine if we need to write header
        file_exists = os.path.exists(log_filename)
        
        with open(log_filename, "a", encoding="utf-8") as f:
            if not file_exists:
                # Write header for new file
                f.write("GPU USAGE MONITORING\n")
                f.write("=" * 50 + "\n")
                f.write(f"Audio File: {audio_filename}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write("=" * 50 + "\n\n")
            
            # Log the current stage and GPU usage
            f.write(f"Stage: {stage}\n")
            if gpu_usage:
                f.write(f"  Total GPU Memory: {gpu_usage.get('nvidia_smi_total_memory_gb', 'N/A')} GB\n")
                f.write(f"  Used Memory: {gpu_usage.get('nvidia_smi_used_memory_gb', 'N/A')} GB\n")
                f.write(f"  Memory Usage: {gpu_usage.get('nvidia_smi_memory_percent', 'N/A'):.2f}%\n")
                f.write(f"  GPU Compute Utilization: {gpu_usage.get('nvidia_smi_gpu_utilization', 'N/A')}%\n")
                f.write(f"  Memory Utilization: {gpu_usage.get('nvidia_smi_memory_utilization', 'N/A')}%\n")
            else:
                f.write("  GPU not available\n")
            
            if additional_info:
                f.write(f"  Additional Info: {additional_info}\n")
            
            f.write(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 30 + "\n\n")
