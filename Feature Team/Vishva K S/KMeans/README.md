# Diarization using KMeans 
# Speaker Segmentation with Transcript Alignment

This Python script performs speaker segmentation on an audio file and aligns the resulting segments with provided transcripts. The speaker segmentation is based on clustering of Mel-frequency cepstral coefficients (MFCCs) using the KMeans algorithm.

## Requirements
- Python 3.x
- Librosa library: `pip install librosa`
- scikit-learn library: `pip install scikit-learn`

## Usage
1. Place the audio file in the same directory as the script or provide the full path to the audio file.
2. Run the script.
3. Ensure you have a list of transcripts corresponding to the audio file. The transcripts should align with the speaker segments.
4. Provide the list of transcripts in the `transcripts` variable.
5. Run the script again.

## Instructions
- Adjust the number of clusters (`n_clusters`) in the KMeans algorithm based on the expected number of speakers in the audio.
- Ensure that the transcripts list has at least as many elements as there are speaker segments detected in the audio. If transcripts are missing for some segments, handle this case accordingly.
- Adjust the threshold for combining segments belonging to the same speaker (`threshold`) based on your requirements.

