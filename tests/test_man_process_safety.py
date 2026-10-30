import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from em_skillopt.core.artifacts import RunArtifacts
from em_skillopt.core.types import EpisodeSpec, EpisodeResult
from em_skillopt.skillopt.rollout import _process_rollout, rollout_specs, RolloutConfig


class Connection:
    def __init__(self, buffer): self.buffer=buffer; self.terminal=False
    def send(self, message): self.buffer.append(message)
    def recv(self):
        if self.terminal: raise AssertionError('Parent waited for EOF after complete shard')
        if not self.buffer: raise EOFError
        value=self.buffer.pop(0)
        if value[0]=='ok': self.terminal=True
        return value
    def close(self): pass


class Process:
    def __init__(self, target, args, daemon): self.target,self.args=target,args
    def start(self): self.target(*self.args)
    def join(self, timeout=None): pass
    def is_alive(self): return False


class Context:
    Process=Process
    def Pipe(self, duplex):
        buffer=[]
        return Connection(buffer),Connection(buffer)


class ManProcessSafety(unittest.TestCase):
    def test_child_saved_results_returned_and_images_isolated(self):
        specs=[EpisodeSpec(f'eb-man:base:{i:04d}','eb-man',i,'task',capability='base') for i in range(4)]
        with tempfile.TemporaryDirectory() as root:
            artifacts=RunArtifacts(root); paths=[]
            def worker(worker_spec, shard, conn):
                self.assertEqual(worker_spec.out_dir,root)
                paths.append(worker_spec.image_dir)
                rows=[]
                for spec in shard:
                    result=EpisodeResult(spec=spec,success=False)
                    artifacts.save_episode(result)
                    rows.append(result)
                conn.send(('ok',rows))
            def factory(cap): return SimpleNamespace(out_dir='WRONG',image_dir='COLLISION',skill=None)
            with patch('em_skillopt.skillopt.rollout.mp.get_context',return_value=Context()), patch('em_skillopt.skillopt.rollout._spawn_entry',side_effect=worker):
                rows=_process_rollout(specs,None,artifacts=artifacts,proc_spec_factory=factory,config=RolloutConfig(workers=2,resume=True))
            self.assertEqual(len(rows),4)
            self.assertEqual(len(set(paths)),2)

    def test_saved_results_recovered_and_included_in_summary(self):
        specs=[EpisodeSpec('eb-man:base:0000','eb-man',0,'task')]
        with tempfile.TemporaryDirectory() as root:
            artifacts=RunArtifacts(root)
            artifacts.save_json('rollout_config.json', {'env_name':'eb-man','split':specs[0].split,'skill_sha256':''})
            artifacts.save_episode(EpisodeResult(spec=specs[0],success=True))
            with patch('em_skillopt.skillopt.rollout._process_rollout',return_value=[]):
                rows=rollout_specs(specs,None,out_dir=root,runner_factory=None,proc_spec_factory=lambda c:None,config=RolloutConfig(workers=2,resume=False))
            self.assertEqual(len(rows),1)
            self.assertTrue(rows[0].success)
            self.assertEqual(artifacts.load_json('summary.json')['total'],1)
            with self.assertRaisesRegex(RuntimeError,'fingerprint mismatch'):
                rollout_specs(specs,'different skill',out_dir=root,runner_factory=None,proc_spec_factory=lambda c:None,config=RolloutConfig(workers=2))

    def test_incomplete_man_batch_cannot_reach_summary_or_gate(self):
        specs=[EpisodeSpec('eb-man:base:0000','eb-man',0,'task')]
        with tempfile.TemporaryDirectory() as root:
            with patch('em_skillopt.skillopt.rollout._process_rollout',return_value=[]):
                with self.assertRaisesRegex(RuntimeError,'Incomplete MAN batch'):
                    rollout_specs(specs,None,out_dir=root,runner_factory=None,proc_spec_factory=lambda c:None,config=RolloutConfig(workers=2,resume=False))


if __name__=='__main__': unittest.main()
