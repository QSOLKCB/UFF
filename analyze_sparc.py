#!/usr/bin/env python3
"""Backward-compatible launcher for the UFF v4 command line.

Existing invocations such as ``python analyze_sparc.py --csv ... --gal ...``
continue to work.  New code may prefer ``python -m uff fit ...``.
"""

from uff.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
