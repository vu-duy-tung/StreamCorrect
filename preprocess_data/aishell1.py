#!/usr/bin/env python3
"""
Small utility to convert two-column AIShell-style txt (id + tokenized text)
into a JSON array of objects with `audio_path` and `text_zh` keys.

Usage:
	python preprocess_data/aishell1.py input.txt output.json --ext .wav

"""
import os
import argparse
import json
import sys
from pathlib import Path


def parse_lines(lines, ext=None, prefix=None):
	out = []
	limit = 0
	for lineno, line in enumerate(lines, start=1):
		line = line.strip()
		if not line:
			continue
		parts = line.split(maxsplit=1)
		if len(parts) == 1:
			id_ = parts[0]
			text = ""
		else:
			id_, text = parts
		audio = id_ + ext if ext else id_
		if prefix:
			prefix_ = os.path.join(prefix, id_[6:11])
			audio = str(Path(prefix_) / audio)

		out.append({"audio_path": audio, "text_zh": text})
		limit += 1
		# if limit >= 10:
		# 	break
	return out


def main():
	p = argparse.ArgumentParser()
	p.add_argument("infile", help="Input txt file")
	p.add_argument("outfile", help="Output json file")
	p.add_argument("--ext", default=None, help="Optional audio extension to append, e.g., .wav")
	p.add_argument("--prefix", default=None, help="Optional path prefix to prepend to audio_path entries")
	args = p.parse_args()

	try:
		with open(args.infile, "r", encoding="utf-8") as f:
			lines = f.readlines()
	except Exception as e:
		print(f"Failed to read {args.infile}: {e}", file=sys.stderr)
		sys.exit(2)

	data = parse_lines(lines, ext=args.ext, prefix=args.prefix)

	try:
		with open(args.outfile, "w", encoding="utf-8") as f:
			json.dump(data, f, ensure_ascii=False, indent=2)
	except Exception as e:
		print(f"Failed to write {args.outfile}: {e}", file=sys.stderr)
		sys.exit(3)


if __name__ == "__main__":
	main()

