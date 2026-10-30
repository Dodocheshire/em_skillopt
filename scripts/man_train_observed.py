"""Launch unchanged MAN training with passive diagnostic timing and stacks."""
import faulthandler
import json
import os
import runpy
import time

faulthandler.enable()
faulthandler.dump_traceback_later(300, repeat=True)
from em_skillopt.backends.qwen_target import QwenTargetClient

_original_generate = QwenTargetClient.generate

def observed_generate(self, messages, **kwargs):
    started = time.monotonic()
    print('[request-start]', json.dumps({'pid': os.getpid(), 'time': time.time(),
          'stage': kwargs.get('stage'), 'max_tokens': kwargs.get('max_tokens'),
          'messages': len(messages)}), flush=True)
    try:
        result = _original_generate(self, messages, **kwargs)
        print('[request-done]', json.dumps({'pid': os.getpid(),
              'elapsed': time.monotonic()-started, 'finish_reason': result.finish_reason,
              'response_chars': len(result.text)}), flush=True)
        return result
    except Exception as exc:
        print('[request-error]', json.dumps({'pid': os.getpid(),
              'elapsed': time.monotonic()-started, 'error_type': type(exc).__name__}), flush=True)
        raise

QwenTargetClient.generate = observed_generate
try:
    runpy.run_module('em_skillopt.cli.train', run_name='__main__')
finally:
    faulthandler.cancel_dump_traceback_later()
