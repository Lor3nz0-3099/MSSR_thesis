"""Background nvidia-smi sampling spanning application startup and native training.

This module imports no CUDA runtime and can start before AppLauncher. Samples
observe process/driver memory including non-Torch allocations; they are not exact
allocation peaks. Missing or unsupported measurements remain None.
"""
import csv
import math
import os
from pathlib import Path
import subprocess
import threading
import time


class GPUMemoryMonitor:
    def __init__(self, output: Path, interval_s=1., *, query_fn=None, pid=None, gpu_index=0):
        self.output = Path(output)
        self.interval_s = float(interval_s)
        if not math.isfinite(self.interval_s) or self.interval_s <= 0:
            raise ValueError('interval_s must be positive and finite')
        self.pid = os.getpid() if pid is None else int(pid)
        self.gpu_index = int(gpu_index)
        self.query_fn = query_fn or self._query
        self.started = time.monotonic()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._rows = []

    def _query(self):
        def run(arguments):
            return subprocess.run(['nvidia-smi', *arguments, '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=2, check=True).stdout
        gpu = run(['--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu'])
        try:
            process = run(['--query-compute-apps=pid,used_gpu_memory'])
        except (OSError, subprocess.SubprocessError):
            process = None
        return gpu, process

    @staticmethod
    def _number(value):
        try:
            numeric = float(value.strip())
            return numeric if math.isfinite(numeric) and numeric >= 0 else None
        except (ValueError, AttributeError):
            return None

    def sample(self):
        """Take one explicit sample; the background thread calls the same method."""
        row = dict(elapsed_s=time.monotonic()-self.started, gpu_index=self.gpu_index,
            total_vram_bytes=None, global_used_bytes=None, free_bytes=None,
            utilization_percent=None, process_gpu_bytes=None, error=None)
        try:
            gpu_text, process_text = self.query_fn()
            found = False
            for line in gpu_text.splitlines():
                values = next(csv.reader([line]))
                if len(values) < 5 or values[0].strip() != str(self.gpu_index):
                    continue
                found = True
                for field, value in zip(('total_vram_bytes', 'global_used_bytes', 'free_bytes'), values[1:4]):
                    number = self._number(value)
                    row[field] = None if number is None else int(number * 1024**2)
                row['utilization_percent'] = self._number(values[4])
                break
            if not found or row['global_used_bytes'] is None:
                row['error'] = 'GPU memory query returned no supported measurement'
            process_values = []
            if process_text is not None:
                for line in process_text.splitlines():
                    values = next(csv.reader([line]))
                    if len(values) >= 2 and values[0].strip() == str(self.pid):
                        number = self._number(values[1])
                        if number is not None:
                            process_values.append(int(number * 1024**2))
            if process_values:
                row['process_gpu_bytes'] = sum(process_values)
        except Exception as error:
            row['error'] = f'{type(error).__name__}: {error}'
        with self._lock:
            self._rows.append(row)
        return dict(row)

    def start(self):
        if self._thread is not None:
            if self._thread.is_alive():
                return self
            raise RuntimeError('a stopped GPU monitor cannot be restarted')
        self._thread = threading.Thread(target=self._run, name='c1-gpu-memory-monitor', daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            started = time.monotonic()
            self.sample()
            self._stop.wait(max(0., self.interval_s-(time.monotonic()-started)))

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()  # each real subprocess query has a two-second bound
        with self._lock:
            rows = [dict(row) for row in self._rows]
        self.output.parent.mkdir(parents=True, exist_ok=True)
        fields = ('elapsed_s', 'gpu_index', 'total_vram_bytes', 'global_used_bytes',
            'free_bytes', 'utilization_percent', 'process_gpu_bytes', 'error')
        with self.output.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return self.snapshot()

    def snapshot(self):
        with self._lock:
            rows = [dict(row) for row in self._rows]
        def peak(name):
            values = [row[name] for row in rows if row[name] is not None]
            return max(values) if values else None
        def first(name):
            return next((row[name] for row in rows if row[name] is not None), None)
        return dict(sample_count=sum(row['global_used_bytes'] is not None for row in rows),
            query_error_count=sum(row['error'] is not None for row in rows),
            total_vram_bytes=first('total_vram_bytes'),
            global_used_sampled_peak_bytes=peak('global_used_bytes'),
            process_gpu_sampled_peak_bytes=peak('process_gpu_bytes'),
            initial_free_bytes=first('free_bytes'), utilization_sampled_peak_percent=peak('utilization_percent'),
            measurement_interval_s=self.interval_s, gpu_index=self.gpu_index, pid=self.pid,
            output=str(self.output), measurement_note='nvidia-smi sampled peaks at the requested cadence; short allocations between samples may be missed. Global memory includes other processes. Process memory covers all GPUs reported for this PID. Torch allocator peaks are separate.')

    def __enter__(self):
        return self.start()

    def __exit__(self, exception_type, exception, traceback):
        self.stop()
        return False
