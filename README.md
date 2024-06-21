# Audio Transcription and Normalization

This project provides a script to split audio files into chunks, transcribe them using OpenAI's Whisper model, and normalize the transcriptions by removing noise words without altering the main content.

## Prerequisites

- Python 3.7+
- `pydub` library
- `openai` library
- FFmpeg or libav for audio processing

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/zonder/my_transcriber.git
    cd my_transcriber
    ```

2. Install the required Python packages:
    ```bash
    pip install pydub openai
    ```

3. Ensure FFmpeg or libav is installed on your system.

## Usage

1. Set your OpenAI API key:
    ```python
    os.environ['OPENAI_API_KEY'] = 'your_openai_api_key'
    ```

2. Specify the path to your audio file:
    ```python
    AUDIO_FILE_PATH = "path_to_your_file.mp3"
    ```

3. Run the script:
    ```bash
    python script.py
    ```

## Code Overview

- `split_audio(input_file, chunk_duration_ms, output_folder)`: Splits the input audio file into chunks and saves them to the output folder.

- `transcribe(client, audio_file_path)`: Transcribes the given audio file using OpenAI's Whisper model.

- `generate_corrected_transcript(client, audio_file_path)`: Transcribes and normalizes the transcription by removing noise words.

- `finalize_transcription(client, transcription)`: Reviews and ensures the transcriptions are contextually correct.

- Main script execution:
    - Splits the audio file into chunks.
    - Transcribes and normalizes each chunk.
    - Combines the transcriptions into a single output file.

## License

This project is licensed under the MIT License.
