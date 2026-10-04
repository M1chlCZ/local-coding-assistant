"""Readiness checks without CUDA, Docker, Windows, or model downloads."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import learning_bootstrap as bootstrap
from train_adapter import atomic_json


class BootTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.child=self.root/'research';self.child.mkdir()
        self.controller=self.root/'continuous';self.controller.mkdir()
        self.model=self.root/'mounted-drive'/'teacher.gguf'
        atomic_json(self.child/'status.json',{'teacher_model':str(self.model),'status':'running'})
        atomic_json(self.controller/'state.json',{'child':str(self.child)})
        atomic_json(self.controller/'command.json',{'action':'resume'})

    def tearDown(self):self.temp.cleanup()

    def test_unavailable_model_drive_retries_before_starting_a_worker(self):
        with patch.object(bootstrap,'docker_ready') as docker:
            ready,reason=bootstrap.check_ready(self.controller,continuous=True)
        self.assertFalse(ready);self.assertIn('model drive',reason)
        docker.assert_not_called()

    def test_new_namespace_can_become_ready_without_changing_saved_control(self):
        original=(self.controller/'command.json').read_bytes()
        self.assertFalse(bootstrap.check_ready(self.controller,continuous=True)[0])
        self.model.parent.mkdir();self.model.write_bytes(b'mounted')
        with patch.object(bootstrap,'docker_ready',return_value=True):
            self.assertTrue(bootstrap.check_ready(self.controller,continuous=True)[0])
        self.assertEqual((self.controller/'command.json').read_bytes(),original)

    def test_docker_boot_delay_is_retryable(self):
        self.model.parent.mkdir();self.model.write_bytes(b'mounted')
        with patch.object(bootstrap,'docker_ready',return_value=False):
            ready,reason=bootstrap.check_ready(self.controller,continuous=True)
        self.assertFalse(ready);self.assertIn('test runtime',reason)

    def test_paused_login_requires_drive_readiness_but_never_resumes_learning(self):
        atomic_json(self.controller/'command.json',{'action':'pause'})
        original=(self.controller/'command.json').read_bytes()
        self.assertFalse(bootstrap.check_ready(self.controller,continuous=True)[0])
        self.model.parent.mkdir();self.model.write_bytes(b'mounted')
        with patch.object(bootstrap,'docker_ready',return_value=False):
            self.assertTrue(bootstrap.check_ready(self.controller,continuous=True)[0])
        self.assertEqual((self.controller/'command.json').read_bytes(),original)

    def test_stopped_controller_does_not_need_a_missing_model(self):
        atomic_json(self.controller/'command.json',{'action':'stop'})
        self.assertTrue(bootstrap.check_ready(self.controller,continuous=True)[0])


if __name__=='__main__':unittest.main()
