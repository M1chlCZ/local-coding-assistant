import concurrent.futures
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import launcher

if __name__ == '__main__':
    config = json.loads((launcher.ROOT / 'research/source_manifest.json').read_text())
    target = launcher.CACHE / 'research/base'
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda artifact: launcher.download(artifact, target), config['artifacts']))
    print('Original source weights and configuration verified.', flush=True)
