"""Compat shim to allow running `python main.py` directly."""

from gemba.cli import run

if __name__ == "__main__":
    run()
