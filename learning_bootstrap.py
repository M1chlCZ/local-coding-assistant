"""Check boot dependencies in a fresh WSL process before starting long-lived learning."""
import argparse
import json
import subprocess
from pathlib import Path


RETRY_EXIT = 75


def docker_ready():
    try:
        return subprocess.run(['docker','info','--format','{{.ServerVersion}}'],
            capture_output=True,timeout=5).returncode==0
    except (OSError,subprocess.TimeoutExpired):
        return False


def check_ready(session,continuous=False):
    session=Path(session)
    command=json.loads((session/'command.json').read_text()) if (session/'command.json').exists() else {'action':'resume'}
    if command['action']=='stop':return True,'Saved Stop respected'
    child=session
    if continuous:
        child=Path(json.loads((session/'state.json').read_text())['child'])
    status=json.loads((child/'status.json').read_text())
    if not Path(status['teacher_model']).is_file():
        return False,'Waiting for the Windows model drive in this WSL process'
    if command['action']!='pause' and not docker_ready():
        return False,'Waiting for the isolated test runtime to start'
    return True,'Drive and required services are ready; saved control unchanged'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--continuous',action='store_true')
    args=parser.parse_args()
    try:ready,reason=check_ready(args.session,args.continuous)
    except (OSError,ValueError,KeyError) as error:parser.exit(1,str(error)+'\n')
    print(json.dumps({'ready':ready,'reason':reason}),flush=True)
    if not ready:raise SystemExit(RETRY_EXIT)


if __name__=='__main__':main()
