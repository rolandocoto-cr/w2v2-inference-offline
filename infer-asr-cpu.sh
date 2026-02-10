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

# Convert to WAV, make mono and downgrade quality
outputWav="$tempFolderPath"/tempAudio.wav
apptainer exec --nv --mount type=bind,src="$tempRoot",dst="$tempRoot" asr-silero-python309.sif ffmpeg -y -i "$pathAudiofile" -acodec pcm_u8 -ac 1 -ar 16000 "$outputWav"

# Find the time locations where the audio file should be split
apptainer exec --nv --mount type=bind,src="$tempRoot",dst="$tempRoot" asr-silero-python309.sif python infer-w2v2-find-audio-splits.py "$outputWav" "$intervalsPath" "$transcriptionFile"

# Split the big audio file into small pieces
apptainer exec --nv --mount type=bind,src="$tempRoot",dst="$tempRoot" asr-silero-python309.sif ./ffmpeg-split.sh "$outputWav" "$intervalsPath"

# Transcribe the files
apptainer exec --nv --mount type=bind,src="$tempRoot",dst="$tempRoot" asr-w2v2-lm-python310.sif python infer-w2v2-transcribe-cpu.py cim "$pathAudiofile" "$tempFolderPath" "$modelPath" "$checkpoint" "$lmPath" "$logFile"

# Put the output where you need me to
outputPath=$(cat "$tempFolderPath"/output-filename.txt)
cp "$outputPath" "$outputFile"