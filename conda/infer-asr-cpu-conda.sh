#!/bin/bash

# Input arguments
pathAudiofile="$1"
transcriptionFile="$2"
tempRoot="$3"
outputFile="$4"
logFile="$5"

# Start log file
echo "202" > "$logFile"

# Variables
modelPath="models/cim-241mins-202502"
checkpoint="checkpoint-12000"
lmPath="models/cim-241mins-202502/cim-241mins-01-lm-4-correct.arpa"

# Create folder that will contain the split files
python infer-w2v2-create-folder-name.py "$pathAudiofile" "$tempRoot"
tempFolderPath=$(cat tempFolder.txt)
intervalsPath="$tempFolderPath"/outputIntervals.tsv
mkdir "$tempFolderPath"

echo "$intervalsPath"

# Start the Silero-VAD environment
source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda init
conda activate asr-silero

# Convert to WAV, make mono and downgrade quality
outputWav="$tempFolderPath"/tempAudio.wav
ffmpeg -y -i "$pathAudiofile" -acodec pcm_u8 -ac 1 -ar 16000 "$outputWav"

# Find the time locations where the audio file should be split
python infer-w2v2-find-audio-splits.py "$outputWav" "$intervalsPath" "$transcriptionFile"

# Split the big audio file into small pieces
./ffmpeg-split.sh "$outputWav" "$intervalsPath"

# Start the W2V2 environment
conda deactivate
conda init
conda activate asr-w2v2-lm-py310

# Transcribe the files
python infer-w2v2-transcribe-cpu.py cim "$pathAudiofile" "$tempFolderPath" "$modelPath" "$checkpoint" "$lmPath" "$logFile"

# Put the output where you need me to
outputPath=$(cat "$tempFolderPath"/output-filename.txt)
cp "$outputPath" "$outputFile"

conda deactivate