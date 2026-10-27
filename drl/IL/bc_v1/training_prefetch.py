"""Optional bounded CPU process producer over immutable TRAIN/VAL G2.

The pipe carries the existing lossless sidecar codec, never CUDA tensors.
Credits bound outstanding batches; the child owns its SQLite/cache. Input
batches must be precomputed, so speculative reads cannot advance a sampler.
"""
from collections.abc import Sequence
import multiprocessing as mp
import resource
import time
import traceback

from .sidecar import encode, decode


def _produce(send, stop, credits, batches, root, split, kwargs, cpu_threads, warm_indices):
    import torch
    from .training_data import TrainingDataset
    from .batching import collate_policy_samples
    torch.set_num_threads(cpu_threads)
    try:
        with TrainingDataset(root, split, **kwargs) as data:
            for index in warm_indices:
                if stop.is_set():return
                data.get_for_collate(index)
            for indices in batches:
                while not stop.is_set():
                    if credits.acquire(timeout=.05):
                        break
                if stop.is_set():
                    return
                batch = collate_policy_samples([data.get_for_collate(i) for i in indices])
                send.send_bytes(encode({'kind':'batch', 'batch':batch}))
            send.send_bytes(encode({'kind':'end', 'cache':data.stats(),
                'child_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}))
    except (BrokenPipeError, EOFError):
        pass  # Consumer closed its owned read endpoint during cancellation.
    except BaseException as exc:
        try:
            send.send_bytes(encode({'kind':'error', 'error':repr(exc), 'traceback':traceback.format_exc()}))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        send.close()


class OrderedProcessPrefetch:
    """FIFO, depth-one/two producer. Always use a context manager or close()."""
    def __init__(self, index_batches, root, split, *, expected_identity,
                 dynamic_budget_bytes, template_budget_bytes, depth=2, cpu_threads=1,
                 warm_indices=()):
        if split not in ('train','validation'):
            raise ValueError('Process producer forbids TEST before starting a child')
        if not isinstance(index_batches,Sequence):
            raise TypeError('Process producer requires a precomputed Sequence')
        if type(depth) is not int or not 1<=depth<=2:
            raise ValueError('Bounded process depth must be one or two')
        if type(cpu_threads) is not int or cpu_threads<1:
            raise ValueError('Positive CPU thread count required')
        if any(type(v) is not int or v<0 for v in (dynamic_budget_bytes,template_budget_bytes)):
            raise ValueError('Nonnegative cache byte budgets required')
        batches=tuple(tuple(indices) for indices in index_batches)
        self._context=mp.get_context('spawn')
        self._read,send=self._context.Pipe(duplex=False)
        self._stop=self._context.Event();self._credits=self._context.BoundedSemaphore(depth)
        self._closed=False;self._start=time.monotonic()
        self._stats={'batches':0,'serialized_bytes':0,'startup_s':None,'cache':None,'child_peak_rss_bytes':None}
        kwargs=dict(expected_identity=expected_identity,dynamic_budget_bytes=dynamic_budget_bytes,
                    template_budget_bytes=template_budget_bytes)
        self.process=self._context.Process(target=_produce,
            args=(send,self._stop,self._credits,batches,root,split,kwargs,cpu_threads,tuple(warm_indices)),
            name='bc-g2-process-producer',daemon=True)
        self.process.start()
        send.close()  # Abrupt child death must produce EOF, not a hidden writer.

    @property
    def worker_alive(self):
        return self.process.is_alive()

    def __iter__(self):return self

    def __next__(self):
        if self._closed:raise StopIteration
        while not self._read.poll(.1):
            if not self.process.is_alive():
                code=self.process.exitcode;self.close()
                raise RuntimeError(f'CPU producer exited without completion (exit={code})')
        try:
            payload=self._read.recv_bytes()
        except (EOFError,OSError) as exc:
            code=self.process.exitcode;self.close()
            raise RuntimeError(f'CPU producer died before completion (exit={code})') from exc
        message=decode(payload)
        if self._stats['startup_s'] is None:self._stats['startup_s']=time.monotonic()-self._start
        if message['kind']=='batch':
            self._credits.release()
            self._stats['batches']+=1;self._stats['serialized_bytes']+=len(payload)
            return message['batch']
        if message['kind']=='end':
            self._stats['cache']=message['cache']
            self._stats['child_peak_rss_bytes']=message['child_peak_rss_bytes']
            self.close();raise StopIteration
        self.close()
        raise RuntimeError('CPU producer failed: '+message.get('error','unknown error')+'\n'+message.get('traceback',''))

    def stats(self):return dict(self._stats)

    def close(self):
        if not self._closed:
            self._stop.set();self._read.close()
            self.process.join(timeout=10.)
            if self.process.is_alive():
                self.process.terminate()  # Only this object's own child.
                self.process.join(timeout=3.)
            if self.process.is_alive():raise RuntimeError('Owned CPU producer did not stop')
            self._closed=True

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
