"""Small checks for private history exports and publication protection."""
import json
import tempfile
from pathlib import Path

from check_public_data import findings, private_findings
from history_data import candidates, export_codex, inspect_export, sanitize


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root/'sessions'
        source.mkdir()
        private = root/'private-data/history'
        records = [
            {'type':'session_meta','payload':{'id':'sample','cwd':'/example/repository','git':{'repository_url':'https://example.com/repository'}}},
            {'type':'response_item','payload':{'type':'message','role':'developer','content':[{'type':'input_text','text':'SYSTEM_TEXT_MUST_NOT_EXPORT'}]}},
            {'type':'response_item','payload':{'type':'reasoning','text':'REASONING_MUST_NOT_EXPORT'}},
            {'type':'response_item','payload':{'type':'function_call_output','output':'TOOL_SECRET_MUST_NOT_EXPORT'}},
            {'type':'response_item','payload':{'type':'message','role':'user','content':[{'type':'input_text','text':'Fix this Python parser.\n<environment_context>PRIVATE_CONTEXT</environment_context>'}]}},
            *[{'type':'response_item','payload':{'type':'message','role':'assistant','phase':'commentary','content':[{'type':'output_text','text':'I will check step '+str(step)}]}} for step in range(4)],
            {'type':'response_item','payload':{'type':'message','role':'assistant','phase':'final_answer','content':[{'type':'output_text','text':'Use a state machine.\n```python\nprint(1)\n```'}]}},
        ]
        path = source/'rollout.jsonl'
        path.write_text(''.join(json.dumps(row)+'\n' for row in records))
        (source/'duplicate.jsonl').write_bytes(path.read_bytes())
        report = export_codex(source, private)
        assert report['sessions'] == 1 and report['messages'] == 6
        content = (private/'codex.jsonl').read_text()
        assert not any(value in content for value in ['SYSTEM_TEXT_MUST_NOT_EXPORT','REASONING_MUST_NOT_EXPORT','TOOL_SECRET_MUST_NOT_EXPORT','PRIVATE_CONTEXT'])
        result = inspect_export(private)
        assert result['coding_candidates'] == 1 and result['training_rows'] == 0
        assert 'print(1)' in next(candidates(private))['excerpt'], 'Review selected commentary instead of the code answer'
        assert private_findings(b'private-data/history/codex.jsonl', content.encode())
        assert private_findings(b'renamed.jsonl', content.encode())
        assert not private_findings(b'history_data.py', Path('history_data.py').read_bytes())
        try:
            export_codex(source, private)
        except FileExistsError:
            pass
        else:
            raise AssertionError('Existing private export was overwritten')
    user = 'example-user'
    raw = '/Users/'+user+'/project\n'+'C:'+chr(92)+'Users'+chr(92)+user+chr(92)+'project\n'
    raw += 'token=ghp_'+'a'*36+'\npassword="not-a-real-password"\nAuthorization: Bearer '+'x'*30
    raw += '\n'+json.dumps({'password':'not-a-real-json-password'})
    clean = sanitize(raw)
    assert user not in clean and 'not-a-real-password' not in clean and 'x'*30 not in clean
    assert 'not-a-real-json-password' not in clean, 'JSON credential was not redacted'
    assert not findings(clean.encode())
    print('PASS: private export filtering, deduplication, redaction and publication guards')


if __name__ == '__main__':
    main()
