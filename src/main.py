import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_pipeline():
    """Run the full transcription pipeline in order: transcriber -> processor -> analyzer"""
    scripts = [
        "src/transcriber/main.py",
        "src/processor/main.py", 
        "src/analyzer/main.py"
    ]

    for script in scripts:
        script_path = Path(script)
        if not script_path.exists():
            logger.error(f"Script not found: {script}")
            return

        logger.info(f"Running {script}...")
        try:
            subprocess.run(["uv", "run", script], check=True)
            logger.info(f"Successfully completed {script}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running {script}: {e}")
            return
        except Exception as e:
            logger.error(f"Unexpected error running {script}: {e}")
            return

    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    run_pipeline()
