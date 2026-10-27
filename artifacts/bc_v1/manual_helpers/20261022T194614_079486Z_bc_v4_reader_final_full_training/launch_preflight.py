"""Pinned production admission only: JSON/source hashes, no models or samples."""
from pathlib import Path
import hashlib, json, subprocess, sys
ROOT = Path('/home/lorenzo/MSSR_thesis')
A = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
G2 = ROOT / 'artifacts/bc_v1/sidecar/g2_v4_actor_visible_r1m'
F2 = ROOT / 'artifacts/bc_v1/normalization/f2_actor_visible_r1m_v1'
CONFIG = ROOT / 'configs/bc_v1/training_action_set_v1.json'
EXPECTED = dict(manifest='d80b0774ed1793dedfa101cbaa82e45b7cb460384bfdd36a4291a592e9d97a3f', visibility='e6b819c2e23e1713f70649d83b547d7701d78ef8c0ec7a21668c803c7c8f6234', f2='bdb5304cd620faa7461c448794044af4757b85e77e672ca7022a1083da9f37cd')

def verify():
    if not subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT).decode().strip() == 'wip/t8-c05-human-teleop-20260921':
        raise ValueError('Branch changed')
    if not subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip() == 'fcbc88ec10416a98c12f33553212c55fb9a0a3a1':
        raise ValueError('HEAD changed')
    # Inspect argv tokens, not a broad match on shell source text.
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit(): continue
        try:
            argv = (directory/'cmdline').read_bytes().decode().split('\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for i, token in enumerate(argv[:-1]):
            if token == '-m' and argv[i+1].startswith('drl.IL.bc_v1.train'):
                raise ValueError('Competing BC training process: PID '+directory.name)
    tensorization_sha = hashlib.sha256((ROOT/'drl/IL/bc_v1/tensorization.py').read_bytes()).hexdigest()
    if tensorization_sha != '19f0827e299b2c55598bea9456a32e17013cf939df732bdaf6b52f5605eae93d':
        raise ValueError('Optimized tensorization changed')
    reader_sha = hashlib.sha256((ROOT/'drl/IL/bc_v1/training_data.py').read_bytes()).hexdigest()
    if reader_sha != '9e9b1cc4d54ea901fda457f41961ea62629505797f6d7352f70d430d1f47d362':
        raise ValueError('Accepted reader changed')
    pins = json.loads((A / 'launch_pins.json').read_text())
    for (name, digest) in pins.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen training preparation file changed: ' + name)
    from drl.IL.bc_v1.train_action_set import preflight, load_action_set_training_config
    config = load_action_set_training_config(CONFIG)
    if not config == dict(schema_version='mssr.bc_v1.training_action_set.v1', model_contract='action_set_v1', initialization='scratch', seed=20261021, lr=0.0003, weight_decay=0.0001, batch_size=64, max_epochs=30, patience=8, min_delta=0.002, threshold_logit=0.0):
        raise ValueError('Pinned production training admission mismatch')
    (report, *_) = preflight(G2, config, normalizer_dir=F2)
    if not (report['source'] == str(G2) and report['normalizer_directory'] == str(F2)):
        raise ValueError('Pinned production training admission mismatch')
    if not report['source_manifest_fingerprint'] == EXPECTED['manifest']:
        raise ValueError('Pinned production training admission mismatch')
    if not report['actor_visibility']['fingerprint'] == EXPECTED['visibility']:
        raise ValueError('Pinned production training admission mismatch')
    if not report['identity']['source_identity']['f2_fingerprint'] == EXPECTED['f2']:
        raise ValueError('Pinned production training admission mismatch')
    if not report['splits'] == {'train': {'enabled': True, 'samples': 78302}, 'validation': {'enabled': True, 'samples': 22762}, 'test': {'enabled': False, 'samples': 0}}:
        raise ValueError('Pinned production training admission mismatch')
    if not report['sampler_quotas'] == {'ordinary': 48, 'structural': 16}:
        raise ValueError('Pinned production training admission mismatch')
    if not report['dropout'] == {'h1': 0.1, 'h2': 0.1}:
        raise ValueError('Pinned production training admission mismatch')
    if not report['runtime'] == dict(cpu_threads=1, cpu_interop_threads=1, numeric_mode='FP32 eager', tf32=False, amp=False, compile=False, prefetch=False, sidecar_template_cache_entries=0, cache_budgets_bytes=dict(train_dynamic=2 * 1024 ** 3, train_template=4 * 1024 ** 3, validation_dynamic=256 * 1024 ** 2, validation_template=512 * 1024 ** 2)):
        raise ValueError('Pinned production training admission mismatch')
    report['launch_preflight'] = 'PASS'
    report['accepted_reader_sha256'] = reader_sha
    report['optimized_tensorization_sha256'] = tensorization_sha
    report['competing_BC_training'] = False
    report['optimizer_steps'] = 0
    return report
if __name__ == '__main__':
    print(json.dumps(verify(), sort_keys=True, allow_nan=False))
