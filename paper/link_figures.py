from pathlib import Path
import shutil
import argparse


def link_figures(src_dir, dst_dir):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    for src_file in src_dir.rglob('*.pdf'):
        dst_name = str(src_file.relative_to(src_dir)).replace('/', '__').replace('\\', '__')
        dst_file = dst_dir / dst_name
        # dst_file.symlink_to(src_file)
        shutil.copy(src_file, dst_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('src_dir', type=Path)
    parser.add_argument('dst_dir', type=Path)
    args = parser.parse_args()

    link_figures(args.src_dir, args.dst_dir)


if __name__ == '__main__':
    main()