from pydub import AudioSegment
from pydub.silence import split_on_silence

def remove_background_noise(input_file, output_file, silence_threshold=-40):
    # Load the audio file
    audio = AudioSegment.from_file(input_file)

    # Split the audio on silence
    segments = split_on_silence(audio, silence_thresh=silence_threshold)

    # Concatenate non-silent segments to create the cleaned audio
    cleaned_audio = AudioSegment.silent(duration=0)
    for segment in segments:
        cleaned_audio += segment

    # Export the cleaned audio to a new file
    cleaned_audio.export(output_file, format="wav")

def main():
    # Replace 'path/to/your/input_audio.wav' with the path to your input audio file
    input_file_path = r'C:\Users\thamizh\Desktop\RBG\noise remover\seg_105.wav'
    
    # Replace 'path/to/your/output_cleaned_audio.wav' with your desired output file path
    output_file_path = r'C:\Users\thamizh\Desktop\RBG\noise remover\cleaned_audio.wav'

    # Adjust the silence_threshold if needed (in dB, lower values are more aggressive)
    silence_threshold = -40

    remove_background_noise(input_file_path, output_file_path, silence_threshold)

    print(f"Background noise removed. Cleaned audio saved to: {output_file_path}")

if __name__ == "__main__":
    main()
