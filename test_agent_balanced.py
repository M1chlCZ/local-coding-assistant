import unittest

from agent_balanced import balanced_rows


class BalancedTests(unittest.TestCase):
    def test_replay_changes_weights_without_changing_executed_targets(self):
        actions = [dict(task_id=f'action-{i}', example_kind='executed_repl_repair') for i in range(317)]
        anchors = [dict(task_id=f'{language}-{i}', language=language, messages=['original target'])
                   for language in ('python', 'go', 'typescript', 'rust', 'dart') for i in range(30)]
        result = balanced_rows(actions, anchors)
        self.assertEqual(result[:317], actions)
        replay = result[317:]
        self.assertEqual(len(replay), 320)
        self.assertEqual(len({r['task_id'] for r in result}), len(result))
        self.assertEqual(len({r['replay_source_id'] for r in replay}), 150)
        originals = {r['task_id']: r for r in anchors}
        for row in replay:
            original = originals[row['replay_source_id']]
            self.assertEqual({k: v for k, v in row.items() if k not in ('task_id', 'replay_source_id')},
                             {k: v for k, v in original.items() if k != 'task_id'})
        with self.assertRaises(ValueError):
            balanced_rows(actions, anchors[:-1])
        with self.assertRaises(ValueError):
            balanced_rows(actions, anchors[:-1]+anchors[:1])


if __name__ == '__main__':
    unittest.main()
