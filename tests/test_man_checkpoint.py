import json,tempfile,unittest
from pathlib import Path
from em_skillopt.skillopt.checkpoint import recover_candidate


class CandidateRecovery(unittest.TestCase):
    def test_exact_candidate_reused_and_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            r=Path(directory);(r/'rollout').mkdir()
            (r/'rollout/rollout_config.json').write_text(json.dumps({'skill_sha256':''}))
            patch={'edits':[{'op':'append','content':'Original candidate'}]}
            (r/'merged_patch.json').write_text(json.dumps({'num_merged_edits':1}))
            (r/'selected_patch.json').write_text(json.dumps(patch))
            (r/'candidate.md').write_text('Original candidate\n')
            self.assertEqual(recover_candidate(r,'',6000)[2].content,'Original candidate')
            with self.assertRaisesRegex(RuntimeError,'source skill mismatch'):
                recover_candidate(r,'different current',6000)
            (r/'candidate.md').write_text('Different candidate')
            with self.assertRaisesRegex(RuntimeError,'differs'):
                recover_candidate(r,'',6000)


if __name__=='__main__':unittest.main()
