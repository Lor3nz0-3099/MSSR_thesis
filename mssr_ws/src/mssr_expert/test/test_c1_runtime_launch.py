"""C1 courses must reach Isaac rather than silently selecting legacy terrain."""
from pathlib import Path
import pytest

SOURCE = (Path(__file__).resolve().parents[1] / 'launch/smores_runtime.launch.py').read_text()


@pytest.mark.parametrize('course', ['marl', 'stairs', 'gap'])
def test_c1_launch_declares_and_forwards_physical_course(course):
    for suffix in ('course', 'split', 'seed'):
        assert f'"phase1_{course}_{suffix}"' in SOURCE
        assert f'"--phase1-{course}-{suffix}"' in SOURCE
