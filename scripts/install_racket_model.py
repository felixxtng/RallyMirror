"""Install the selected racket checkpoint after validating its model-card hash."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[1]


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def install(source, card_path=None, destination=None):
    source = Path(source).resolve()
    card_path = Path(card_path) if card_path else REPO/'models/racket_v006.json'
    card = json.loads(card_path.read_text(encoding='utf-8'))
    destination = Path(destination).resolve() if destination else REPO/card['weights_file']
    expected = card['weights_sha256']
    if not source.is_file() or sha(source) != expected:
        raise ValueError('Checkpoint does not match the selected v006 model-card hash')
    if destination.exists():
        if sha(destination) == expected:
            return destination
        raise FileExistsError('A different model already exists at '+str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix+'.installing')
    if temporary.exists():
        raise FileExistsError('Preserve the existing partial install: '+str(temporary))
    try:
        shutil.copyfile(source, temporary)
        if sha(temporary) != expected:
            raise ValueError('Copied checkpoint failed verification')
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    print('Verified runtime model:', install(args.source))


if __name__ == '__main__':
    main()
