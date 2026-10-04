"""Bounded standard-library programs in a locally built, immutable compiler image."""
import json
import subprocess
import uuid
from pathlib import Path
from evaluation import bounded_run
from training_data import sha256

ROOT=Path(__file__).resolve().parent
LANGUAGES=('python','go','typescript','rust','dart')
FILENAMES={'python':'solution.py','go':'solution.go','typescript':'solution.ts','rust':'solution.rs','dart':'solution.dart'}


def image():
    config=json.loads((ROOT/'.cache/polyglot-runtime.json').read_text())
    if config.get('dockerfile_sha256')!=sha256(ROOT/'research/Polyglot.Dockerfile'):
        raise ValueError('Compiler image source changed; rebuild and verify before continuing')
    value=config['image']
    if not value.startswith('sha256:') or len(value)!=71:
        raise ValueError('Compiler runtime requires an immutable image ID')
    return value


def container_command(interactive=False):
    return ['docker','run','--rm','--network','none','--read-only','--cap-drop=ALL',
        '--security-opt=no-new-privileges','--pids-limit=128','--memory=1g','--cpus=2',
        '--user=65534:65534','--tmpfs','/workspace:rw,exec,nosuid,size=128m,uid=65534,gid=65534',
        '--tmpfs','/tmp:rw,exec,nosuid,size=128m,uid=65534,gid=65534',
        '-e','HOME=/workspace','-e','GOCACHE=/workspace/.gocache','-e','GOPATH=/workspace/go',
        '-e','GOMAXPROCS=2','-e','GO111MODULE=off','-e','GOPROXY=off',
        '-e','DART_SUPPRESS_ANALYTICS=true']+(['-i'] if interactive else ['-d'])+[image()]


def commands(language,filename):
    if language=='python':return [],['python','-I','-S',filename]
    if language=='go' and filename.endswith('_test.go'):return [],['go','test',filename]
    if language=='go':return ['go','build','-o','/workspace/program',filename],['/workspace/program']
    if language=='rust':return ['rustc','--edition=2021',filename,'-o','/workspace/program'],['/workspace/program']
    if language=='typescript':return ['tsc','--target','es2022','--module','commonjs','--typeRoots','/workspace/no-types' if filename=='benchmark.ts' else '/usr/local/lib/node_modules/@types',filename],['node',filename.replace('.ts','.js')]
    if language=='dart':return [],['dart','--disable-dart-dev',filename]
    raise ValueError('Unsupported coding language')


def runner_source(task):
    compile_command,run_command=commands(task['language'],task['filename'])
    cases=task['cases']
    return ('import os,subprocess,shutil\n'
        "os.chdir('/workspace')\n"
        "if not os.path.exists('/workspace/.gocache') and os.path.exists('/opt/go-cache'): shutil.copytree('/opt/go-cache','/workspace/.gocache')\n"
        f'compile_command={compile_command!r}\nrun_command={run_command!r}\n'
        'if compile_command:\n'
        ' compiled=subprocess.run(compile_command,capture_output=True,text=True,timeout=30)\n'
        ' if compiled.returncode: raise AssertionError("Compile failure: "+(compiled.stdout+compiled.stderr)[-3000:])\n'
        f'cases={cases!r}\n'
        'for index,case in enumerate(cases):\n'
        ' result=subprocess.run(run_command,input=case["input"],capture_output=True,text=True,timeout=5)\n'
        ' output=result.stdout\n'
        ' if result.returncode or output.split()!=case["output"].split():\n'
        '  raise AssertionError("Case "+str(index)+" failed; exit "+str(result.returncode)+"; output "+repr(output[:1000])+"; stderr "+result.stderr[-1000:])\n')


def check_program(language,source,cases=None,timeout=60,filename=None):
    filename=filename or FILENAMES[language]
    if Path(filename).name!=filename or len(source)>100000:
        raise ValueError('Invalid native source file')
    marker='POLYGLOT_PASSED_'+uuid.uuid4().hex
    if cases is None:
        compile_command,run_command=commands(language,filename)
        checks=("import subprocess,os,shutil\nos.chdir('/workspace')\n"
            "if os.path.exists('/opt/go-cache'): shutil.copytree('/opt/go-cache','/workspace/.gocache')\n"
            f'commands={[compile_command,run_command] if compile_command else [run_command]!r}\n'
            'for command in commands:\n'
            ' result=subprocess.run(command,capture_output=True,text=True,timeout=30)\n'
            ' if result.returncode: raise AssertionError((result.stdout+result.stderr)[-3000:])\n')
    else:checks=runner_source({'language':language,'filename':filename,'cases':cases})
    program=f"from pathlib import Path\nPath('/workspace/{filename}').write_text({source!r})\n"+checks+f'print({marker!r})\n'
    name='local-polyglot-'+uuid.uuid4().hex
    command=container_command(True);command[2:2]=['--name',name]
    try:
        result=bounded_run(command+['python','-I','-S','-'],program,timeout=timeout)
        if result['exit_code'] in (125,126,127):raise RuntimeError('Compiler container infrastructure failed: '+result['output'])
        return {'passed':result['exit_code']==0 and not result['output_limited'] and result['output'].strip().endswith(marker),'feedback':result['output'][-4096:]}
    except subprocess.TimeoutExpired:
        return {'passed':False,'feedback':'Native compilation/execution timed out'}
    finally:subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=15)


def grade(task,changes):
    if set(changes)-{task['filename']} or task['filename'] not in changes:
        raise ValueError('Native patch must change the registered solution only')
    return check_program(task['language'],changes[task['filename']],task['cases'])
