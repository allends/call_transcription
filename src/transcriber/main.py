import whisper
import logging
from pathlib import Path
from typing import Union, Optional, Dict
import tempfile
from pydub import AudioSegment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioTranscriber:
    def __init__(self, model_size: str = "base"):
        """Initialize the transcriber with a specific Whisper model."""
        logger.info(f"Loading Whisper model: {model_size}")
        try:
            self.model = whisper.load_model(model_size)
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def save_transcription(self, text: str, output_path: Path) -> None:
        """Save transcribed text to a file."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            logger.info(f"Saved transcription to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save transcription: {e}")
            raise
    def transcribe_file(
        self, 
        audio_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        language: Optional[str] = None,
        verbose: bool = False
    ) -> Dict:
        """Transcribe an audio file to text and optionally save to file."""
        audio_path = Path(audio_path)
    
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
        logger.info(f"Starting transcription of: {audio_path}")
    
        try:
            # Convert MP3 to WAV if necessary
            if audio_path.suffix.lower() == '.mp3':
                # Create a temporary directory that won't be deleted immediately
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_wav_path = Path(temp_dir) / 'temp_audio.wav'
                    logger.info("Converting MP3 to WAV format")
                    audio = AudioSegment.from_mp3(str(audio_path))
                    audio.export(str(temp_wav_path), format='wav')
                
                    # Perform transcription
                    options = {"verbose": verbose}
                    if language:
                        options["language"] = language

                    result = self.model.transcribe(str(temp_wav_path), **options)
            else:
                # If it's already a WAV file, transcribe directly
                options = {"verbose": verbose}
                if language:
                    options["language"] = language
                result = self.model.transcribe(str(audio_path), **options)
        
            # Save transcription if output path is provided
            if output_path:
                output_path = Path(output_path)
                self.save_transcription(result["text"], output_path)
        
            logger.info("Transcription completed successfully")
            return result

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        file_pattern: str = "*.mp3",
        language: Optional[str] = None
    ) -> Dict[str, Dict]:
        """Process all audio files in a directory and save transcriptions."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        if not input_dir.is_dir():
            raise NotADirectoryError(f"Input directory not found: {input_dir}")

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        for audio_file in input_dir.glob(file_pattern):
            logger.info(f"Processing file: {audio_file}")
            # Create output path with .txt extension
            output_path = output_dir / f"{audio_file.stem}.txt"
            
            try:
                results[audio_file.name] = self.transcribe_file(
                    audio_file,
                    output_path=output_path,
                    language=language
                )
            except Exception as e:
                logger.error(f"Failed to transcribe {audio_file}: {e}")
                results[audio_file.name] = {"error": str(e)}

        return results


def main():
    """Process MP3 files from resources/ directory and save transcriptions."""
    # Get project root directory (assuming this script is in the root)
    project_root = Path(__file__).parent.parent.parent
    resources_dir = project_root / "resources"
    transcriptions_dir = project_root / "transcriptions"

    # Create a transcriber instance
    transcriber = AudioTranscriber(model_size="base")

    try:
        # Process all MP3 files in the resources directory
        results = transcriber.process_directory(
            input_dir=resources_dir,
            output_dir=transcriptions_dir,
            language="en"  # You can change or make this configurable
        )

        # Print summary
        print("\nTranscription Summary:")
        print("-" * 50)
        for filename, result in results.items():
            if "error" in result:
                print(f"❌ {filename}: Failed - {result['error']}")
            else:
                print(f"✅ {filename}: Successfully transcribed")

    except Exception as e:
        print(f"Process failed: {e}")

if __name__ == "__main__":
    main()
