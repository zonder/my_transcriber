from pydub import AudioSegment
import os
from openai import OpenAI

CHUNK_DURATION_SECONDS = 90
OUTPUT_FOLDER = "output_chunks"

os.environ['OPENAI_API_KEY'] = 'OPENAI_API_KEY'


def split_audio(input_file, chunk_duration_ms, output_folder):
    audio = AudioSegment.from_mp3(input_file)
    file_name = os.path.splitext(os.path.basename(input_file))[0]
    os.makedirs(output_folder, exist_ok=True)

    chunk_start = 0
    chunk_files = []

    while chunk_start < len(audio):
        chunk_end = chunk_start + chunk_duration_ms

        if len(audio) - chunk_start < chunk_duration_ms * 0.5:
            chunk_end = len(audio)

        print(chunk_start)
        print(chunk_end)

        chunk = audio[chunk_start:chunk_end]

        chunk_file_name = f"{file_name}_chunk_{chunk_start // 1000}_{chunk_end // 1000}.mp3"
        chunk_file_path = os.path.join(output_folder, chunk_file_name)
        chunk.export(chunk_file_path, format="mp3")
        chunk_files.append(chunk_file_path)

        chunk_start = chunk_end

    return chunk_files


def transcribe(client, audio_file_path):
    audio_file = open(audio_file_path, "rb")
    print("Transcribing...")
    return client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="text",
        prompt="chatgpt, E-Builder, 3rd party")


def generate_corrected_transcript(client, audio_file_path):
    transcription = transcribe(client, audio_file_path)
    # print(transcription)
    print("Normalizing...")
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "Act as a transcriptionist. Please transcribe the audio file provided. Ensure that all noise words (e.g., 'um', 'ah', 'like') are removed, but do not alter the main content or wording of the transcription. The transcription should be in the original language of the audio file."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    normalized_transcription = completion.choices[0].message.content
    # print("Normalized: " + normalized_transcription)
    return normalized_transcription


def finalize_transcription(client, transciption):
    print("Grooping...")
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "Act as a transcription editor. Please review the transcribed chunks provided. Ensure they are contextually correct, but do not change the text or wording. The review should be in the original language of the transcription."
            },
            {
                "role": "user",
                "content": transciption
            }
        ]
    )
    return completion.choices[0].message.content


AUDIO_FILE_PATH = "PATH_TO_YOUR_FILE"

print("Creating audio chunks...")
chunk_files = split_audio(
    AUDIO_FILE_PATH, CHUNK_DURATION_SECONDS*1000, OUTPUT_FOLDER)

client = OpenAI()
transcription = ""
for chunk_file in chunk_files:
    print(F"Preparing transciption for {chunk_file}")
    transcription += generate_corrected_transcript(client, chunk_file) + "\n"

with open(F"{OUTPUT_FOLDER}\output.txt", 'w', encoding='utf-8') as file:
    file.write(transcription)
