from pathlib import Path

import pytest


def test_monitor_parses_samples_and_stops_thread(tmp_path):
    from drl.MARL.c1_resource_monitor import GPUMemoryMonitor
    responses = iter([
        ('0, 8192, 2000, 6192, 30', '123, 500'),
        ('0, 8192, 3000, 5192, 50', '123, 800'),
    ])
    def query():
        return next(responses)
    monitor = GPUMemoryMonitor(tmp_path / 'gpu.csv', interval_s=.01, query_fn=query, pid=123)
    # Explicit samples exercise deterministic parsing without relying on cadence.
    monitor.sample()
    monitor.sample()
    result = monitor.snapshot()
    assert result['sample_count'] == 2
    assert result['total_vram_bytes'] == 8192 * 1024**2
    assert result['initial_free_bytes'] == 6192 * 1024**2
    assert result['global_used_sampled_peak_bytes'] == 3000 * 1024**2
    assert result['process_gpu_sampled_peak_bytes'] == 800 * 1024**2
    assert 'sampled' in result['measurement_note']
    monitor.stop()
    assert (tmp_path / 'gpu.csv').read_text().count('\n') == 3
    with GPUMemoryMonitor(tmp_path / 'background.csv', interval_s=.01,
            query_fn=lambda: ('0, 8192, 1000, 7192, 5', '')) as running:
        assert running._thread.is_alive()
    assert not running._thread.is_alive()


def test_failed_queries_and_missing_process_data_are_not_zero(tmp_path):
    from drl.MARL.c1_resource_monitor import GPUMemoryMonitor
    def fail():
        raise OSError('missing nvidia-smi')
    monitor = GPUMemoryMonitor(tmp_path / 'gpu.csv', query_fn=fail)
    monitor.sample()
    result = monitor.snapshot()
    assert result['sample_count'] == 0
    assert result['query_error_count'] == 1
    assert result['global_used_sampled_peak_bytes'] is None
    assert result['process_gpu_sampled_peak_bytes'] is None
    monitor.query_fn = lambda: ('0, 8192, 1000, 7192, 5', '')
    monitor.sample()
    result = monitor.snapshot()
    assert result['sample_count'] == 1
    assert result['process_gpu_sampled_peak_bytes'] is None
    monitor.stop()


def test_context_stops_on_body_exception(tmp_path):
    from drl.MARL.c1_resource_monitor import GPUMemoryMonitor
    monitor = GPUMemoryMonitor(tmp_path / 'gpu.csv', interval_s=.01,
        query_fn=lambda: ('0, 8192, 1000, 7192, 5', ''))
    with pytest.raises(RuntimeError):
        with monitor:
            raise RuntimeError('trainer failed')
    assert not monitor._thread.is_alive()
    assert (tmp_path / 'gpu.csv').is_file()
