import re
from pathlib import Path
from typing import Dict, Set, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranscriptionProcessor:
    def __init__(self):
        """Initialize the transcription processor with common filler words and patterns."""
        # Common filler words in conversations
        self.filler_words = {
            'um', 'uh', 'eh', 'ah', 'er', 'like', 'you know', 'i mean', 
            'sort of', 'kind of', 'basically', 'literally', 'actually',
            'so yeah', 'right', 'okay', 'well', 'so', 'anyway', 'anyways',
            'yeah', 'mhm', 'hmm', 'erm'
        }
        
        # Common speech disfluencies and repetitions
        self.disfluencies = {
            r'\b(the the|i i|a a)\b',  # Word repetitions
            r'\b(i mean like|you know like)\b',  # Combined fillers
            r'\b(so basically|like basically)\b'  # Common combinations
        }
        
        # Patterns for standardization
        self.standardization_patterns = {
            r'\b(cant|can\'t)\b': 'cannot',
            r'\b(dont|don\'t)\b': 'do not',
            r'\b(didnt|didn\'t)\b': 'did not',
            r'\b(wouldnt|wouldn\'t)\b': 'would not',
            r'\b(couldnt|couldn\'t)\b': 'could not',
            r'\b(shouldnt|shouldn\'t)\b': 'should not',
            r'\b(wont|won\'t)\b': 'will not',
            r'\b(hasnt|hasn\'t)\b': 'has not',
            r'\b(havent|haven\'t)\b': 'have not',
            r'\b(isnt|isn\'t)\b': 'is not',
            r'\b(arent|aren\'t)\b': 'are not',
            r'\b(wasnt|wasn\'t)\b': 'was not',
            r'\b(werent|weren\'t)\b': 'were not'
        }

    def remove_filler_words(self, text: str) -> str:
        """Remove filler words from the text."""
        # Create a regex pattern for whole word matching of filler words
        pattern = r'\b(' + '|'.join(re.escape(word) for word in self.filler_words) + r')\b'
        return re.sub(pattern, '', text, flags=re.IGNORECASE)

    def remove_disfluencies(self, text: str) -> str:
        """Remove speech disfluencies and repetitions."""
        for pattern in self.disfluencies:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        return text

    def standardize_text(self, text: str) -> str:
        """Standardize contractions and common variations."""
        # Apply standardization patterns
        for pattern, replacement in self.standardization_patterns.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Additional standardization
        text = re.sub(r'\s+', ' ', text)  # Standardize whitespace
        text = text.strip()  # Remove leading/trailing whitespace
        return text

    def clean_transcription(self, text: str) -> str:
        """Apply all cleaning steps to the transcription."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers (preserve basic punctuation)
        text = re.sub(r'[^a-z\s.,!?]', '', text)
        
        # Apply cleaning steps
        text = self.remove_filler_words(text)
        text = self.remove_disfluencies(text)
        text = self.standardize_text(text)
        
        return text

    def process_file(self, input_path: Path, output_path: Optional[Path] = None) -> str:
        """Process a single transcription file."""
        try:
            # Read input file
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # Clean the transcription
            cleaned_text = self.clean_transcription(text)

            # Save to output file if path is provided
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)
                logger.info(f"Processed transcription saved to: {output_path}")

            return cleaned_text

        except Exception as e:
            logger.error(f"Error processing file {input_path}: {e}")
            raise

    def process_directory(
        self,
        input_dir: Path,
        output_dir: Optional[Path] = None,
        file_pattern: str = "*.txt"
    ) -> Dict[str, str]:
        """Process all transcription files in a directory."""
        input_dir = Path(input_dir)
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        for input_file in input_dir.glob(file_pattern):
            try:
                if output_dir:
                    output_file = output_dir / input_file.name
                else:
                    output_file = None

                cleaned_text = self.process_file(input_file, output_file)
                results[input_file.name] = cleaned_text
                logger.info(f"Successfully processed: {input_file.name}")

            except Exception as e:
                logger.error(f"Failed to process {input_file.name}: {e}")
                results[input_file.name] = f"ERROR: {str(e)}"

        return results

    def add_custom_filler_words(self, words: Set[str]) -> None:
        """Add custom filler words to the processor."""
        self.filler_words.update(words)

    def add_custom_patterns(self, patterns: Dict[str, str]) -> None:
        """Add custom standardization patterns to the processor."""
        self.standardization_patterns.update(patterns)


def main():
    """Example usage of the TranscriptionProcessor."""
    processor = TranscriptionProcessor()

    # Get project root directory
    project_root = Path(__file__).parent.parent.parent
    transcriptions_dir = project_root / "transcriptions"
    processed_dir = project_root / "processed_transcriptions"

    try:
        # Process all transcription files
        results = processor.process_directory(
            input_dir=transcriptions_dir,
            output_dir=processed_dir
        )

        # Print summary
        print("\nProcessing Summary:")
        print("-" * 50)
        for filename, result in results.items():
            if result.startswith("ERROR:"):
                print(f"❌ {filename}: {result}")
            else:
                print(f"✅ {filename}: Successfully processed")

    except Exception as e:
        print(f"Process failed: {e}")


if __name__ == "__main__":
    main()
