import unittest
from test_focused_data import sample
import focused_replay as replay

class ReplayTests(unittest.TestCase):
 def test_candidate_cannot_edit_unregistered_source_or_tests(self):
  row=sample()
  self.assertEqual(replay.validate_patch(row,row['patch']),row['patch'])
  for name in ('other.go','count_test.go','../secret.go'):
   with self.assertRaises(ValueError):replay.validate_patch(row,row['patch'].replace('count.go',name))
 def test_only_publisher_images_and_valid_digest(self):
  with self.assertRaises(ValueError):replay.publisher_image('other/thing:latest')
  self.assertEqual(replay.publisher_image('docker.io/swerebenchv2/project-name:3-abcdef'), 'docker.io/swerebenchv2/project-name:3-abcdef')
 def test_gold_and_baseline_control_required(self):
  self.assertTrue(replay.controls_valid({'exit_code':1,'test_started':True,'output_limited':False},{'exit_code':0,'test_started':True,'output_limited':False}))
  self.assertFalse(replay.controls_valid({'exit_code':1,'test_started':False},{'exit_code':0,'test_started':True}))
  self.assertFalse(replay.controls_valid({'exit_code':0,'test_started':True},{'exit_code':0,'test_started':True}))

if __name__=='__main__':unittest.main()
