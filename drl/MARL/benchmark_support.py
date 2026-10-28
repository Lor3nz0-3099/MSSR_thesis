"""Small monotonic wall timers; callback leaf timers may overlap parent spans.

Context spans track exclusive time by subtracting nested context spans. External
callback timings are separately reported and must not be added to parent spans.
No CUDA synchronization is introduced: GPU values measure host submission time.
"""
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
import time


class MonotonicProfile:
    def __init__(self, clock=time.monotonic):
        self.clock=clock
        self.started=clock()
        self.timers=defaultdict(lambda:dict(calls=0,inclusive_s=0.,exclusive_s=0.))
        self.stack=[]
        self.domains=defaultdict(lambda:defaultdict(float))

    def add(self,name,elapsed_seconds):
        row=self.timers[name]
        row['calls']+=1
        row['inclusive_s']+=elapsed_seconds
        row['exclusive_s']+=elapsed_seconds

    @contextmanager
    def measure(self,name):
        frame=[self.clock(),0.]
        self.stack.append(frame)
        try:
            yield
        finally:
            elapsed=self.clock()-frame[0]
            self.stack.pop()
            if self.stack:
                self.stack[-1][1]+=elapsed
            row=self.timers[name]
            row['calls']+=1
            row['inclusive_s']+=elapsed
            row['exclusive_s']+=max(0.,elapsed-frame[1])

    def wrap(self,name,function):
        @wraps(function)
        def measured(*args,**kwargs):
            with self.measure(name):
                return function(*args,**kwargs)
        return measured

    def snapshot(self):
        return {name:dict(row) for name,row in self.timers.items()}

    def observe(self,domain,*,reward,progress,success,conflict,invalid,accepted):
        row=self.domains[domain]
        row['transitions']+=1
        for name,value in dict(reward=reward,progress=progress,success=success,conflict=conflict,
                               invalid=invalid,accepted=accepted).items():
            row[name]+=float(value)

    def domain_summary(self):
        return {domain:dict(transitions=int(row['transitions']),
            **{name+('_mean' if name in ('reward','progress') else '_rate'):row[name]/row['transitions']
               for name in ('reward','progress','success','conflict','invalid','accepted')})
            for domain,row in self.domains.items() if row['transitions']}

    def summary(self,*,transitions,updates,wall_s=None):
        wall=self.clock()-self.started if wall_s is None else wall_s
        return dict(schema='mssr.c1_profile.v1',wall_s=wall,transitions=transitions,updates=updates,
            transitions_per_s=transitions/wall if wall else None,
            seconds_per_transition=wall/transitions if transitions else None,
            updates_per_s=updates/wall if wall else None,
            wall_seconds_per_update=wall/updates if updates else None,
            eta_seconds={str(n):wall*n/transitions if transitions else None for n in (10000,100000,500000)},
            domains=self.domain_summary(),timers=self.snapshot(),
            timing_notes='Context timers: inclusive and nested-exclusive wall; inference/update callback leaves overlap parent spans. Host wall only; no GPU synchronization. Success_rate here is transition success incidence; episode success rates are in result.json.')
