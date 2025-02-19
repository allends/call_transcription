# call_transcription

This uses `uv` to manage the dependencies.

To run the transcription processor:

```bash
uv run src/transcriber/main.py
```

This will process all the files in the `resources` directory and save the results in the `transcriptions` directory.

To run the processor:

```bash
uv run src/processor/main.py
```

This will process all the files in the `transcriptions` directory and save the results in the `processed_transcriptions` directory.

To run the issue analyzer:

```bash
uv run src/analyzer/analyze_issues.py
```

This will analyze all files in the `processed_transcriptions` directory and generate a CSV report in `analysis/issues/` with the current date as part of the filename.

To run the topics analyzer:

```bash
uv run src/analyzer/analyze_topics.py
```

This will analyze all files in the `processed_transcriptions` directory and generate a CSV report in `analysis/topics/` with the current date as part of the filename.
