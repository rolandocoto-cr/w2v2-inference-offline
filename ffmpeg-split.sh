#!/bin/bash

# Ensure a WAV file and segments file are provided as arguments
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <path_to_wav_file> <path_to_tsv_file>"
  exit 1
fi

wav_file="$1"
tsv_file="$2"

echo "$wav_file"
echo "$tsv_file"

# Check if the WAV file exists
if [ ! -f "$wav_file" ]; then
  echo "WAV file not found!"
  exit 1
fi

# Check if the segments file exists
if [ ! -f "$tsv_file" ]; then
  echo "Segments file not found!"
  exit 1
fi

# Read the segments file line by line, using tabs as delimiters
while IFS=$'\t' read -r start end output_filename; do
  # Ignore empty lines
  if [[ -z "$start" || -z "$end" || -z "$output_filename" ]]; then
    continue
  fi

  # Use ffmpeg to extract the specified segment
  ffmpeg -i "$wav_file" -ss "$start" -to "$end" -c copy "$output_filename" < /dev/null

done < "$tsv_file"