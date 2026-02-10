#!/bin/bash

# Input arguments
pathAudiofile="$1"
transcriptionFile="$2"
tempRoot="$3"
outputFile="$4"
logFile="$5"

# Get the directory where this script lives
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
localPath="$(dirname "$SCRIPT_PATH")"
if [ "$localPath" = "." ]; then
    localPath=$(pwd)
fi
echo "$localPath"

# Start log file
echo "202" > "$logFile"

# Variables
modelPath="$localPath""/models/cim-241mins-202502"
checkpoint="$localPath""/models/cim-241mins-202502/checkpoint-12000"
lmPath="$localPath""/models/cim-241mins-202502/cim-241mins-01-lm-4-correct.arpa"

# Create folder that will contain the split files
python3 "$localPath"/infer-w2v2-create-folder-name.py "$pathAudiofile" "$tempRoot"
tempFolderPath=$(cat tempFolder.txt)
intervalsPath="$tempFolderPath"/outputIntervals.tsv
mkdir "$tempFolderPath"

# Convert to WAV, make mono and downgrade quality
outputWav="$tempFolderPath"/tempAudio.wav
docker run --rm -v "$localPath":"$localPath" asr-silero ffmpeg -y -i "$pathAudiofile" -acodec pcm_u8 -ac 1 -ar 16000 "$outputWav"

# Find the time locations where the audio file should be split
docker run --rm -v "$localPath":"$localPath" asr-silero python "$localPath"/infer-w2v2-find-audio-splits.py "$outputWav" "$intervalsPath" "$transcriptionFile"

# Split the big audio file into small pieces
docker run --rm -v "$localPath":"$localPath" asr-silero "$localPath"/ffmpeg-split.sh "$outputWav" "$intervalsPath"

# Transcribe the files
docker run --rm -v "$localPath":"$localPath" asr-w2v2-lm-python310 python "$localPath"/infer-w2v2-transcribe-cpu.py cim "$pathAudiofile" "$tempFolderPath" "$modelPath" "$checkpoint" "$lmPath" "$logFile"

# Put the output where you need me to
outputPath=$(cat "$tempFolderPath"/output-filename.txt)
cp "$outputPath" "$outputFile"

if [ -f "$outputFile" ]; then
	echo "200" > "$logFile"
else
	echo "404" > "$logFile"
fi