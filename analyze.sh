#!/bin/bash

uv run src/transcriber/main.py
uv run src/processor/main.py
uv run src/analyzer/main.py

