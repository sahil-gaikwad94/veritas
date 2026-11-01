"""Phase 3 tests: CDCL domain cut."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from veritas.cdcl_jumpback import CdclJumpbackEngine, ClauseCheckpoint


def test_culprit_selection_and_jumpback():
    eng = CdclJumpbackEngine()
    cp1 = ClauseCheckpoint(clause_id=1, var_id="w1", committed_val=0.2, start_token_idx=5, end_token_idx=6, mean_shannon_entropy=2.0, token_sequence=[1, 2], conflict_count=1)
    cp2 = ClauseCheckpoint(clause_id=2, var_id="w2", committed_val=0.3, start_token_idx=7, end_token_idx=8, mean_shannon_entropy=1.0, token_sequence=[3, 4], conflict_count=0)
    eng.push_checkpoint(cp1)
    eng.push_checkpoint(cp2)
    res = eng.jumpback([1, 2])
    assert res["target_clause"] in (1, 2)
    assert res["rewind_to"] is not None
    assert len(eng.tabu.entries) >= 1


def test_domain_cut_record():
    eng = CdclJumpbackEngine()
    cp = ClauseCheckpoint(clause_id=1, var_id="x", committed_val=0.5, start_token_idx=2, end_token_idx=3, mean_shannon_entropy=1.0, token_sequence=[9], conflict_count=2)
    eng.push_checkpoint(cp)
    eng.jumpback([1])
    assert len(eng.domain_cuts) == 1
    assert eng.domain_cuts[0]["var_id"] == "x"
    assert len(eng.export_records()) == 1
