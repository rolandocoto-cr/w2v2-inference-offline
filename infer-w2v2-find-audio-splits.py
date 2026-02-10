#===========================================================
# Find the points in an audio file where there is human
# speech, and determine its start and end point in seconds.
# This uses sound features (e.g. differences in decibels),
# not machine learning.
# This is part of the code for CIM ASR.
#
# Rolando Coto. rolando.a.coto.solano@dartmouth.edu
# Last update: February 18, 2025
#===========================================================

import sys
import os
import csv

from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
vadmodel = load_silero_vad()

from datetime import datetime
import re
import glob
from decimal import Decimal


outputWav = sys.argv[1]
intervalsPath = sys.argv[2]
premadeTSV = sys.argv[3]

print("intervalsPath: " + intervalsPath)
print("premadeTSV:    " + premadeTSV)


def addZerosInFrontOfNumber(number, total):

  lenNum = len(str(number))
  lenTotal = len(str(total))

  zerosToAdd = lenTotal-lenNum

  stringZeros = ""
  for i in range(0,zerosToAdd): stringZeros = stringZeros + "0"
  retNum = stringZeros + str(number)
  return retNum


def fixIntervals(audio_path, file_path):
    # This list will hold all the intervals after processing
    intervals = []

    totalLines = 0

    # Open the file and read lines
    with open(file_path, 'r') as f:
        for line in f:
            totalLines = totalLines + 1
            parts = line.strip().split(",")
            if len(parts) != 2:
                continue
            start, end = float(parts[0]), float(parts[1])

            # Check if the duration exceeds 15 seconds
            while end - start > 15.0:
                # Add an interval with a maximum of 15 seconds
                intervals.append((start, start + 15.0))
                start += 15.0 + 0.001  # Increment start for the next interval

            # Add the final interval (which is <= 15 seconds)
            intervals.append((start, end))

    # Sort intervals based on the start time
    intervals.sort()

    output = ""

    lineCounter = 0
    filenames = []
    # Print the processed and sorted intervals
    for interval in intervals:
        lineCounter = lineCounter + 1
        nameAudio = audio_path.replace(".wav", "-" + str(addZerosInFrontOfNumber(lineCounter,totalLines)) + ".wav")
        filenames.append(nameAudio)
        #output = output + str(round(Decimal(interval[0]),3)) + "\t" + interval[1] + "\n"
        output = output + str(round(Decimal(interval[0]),3)) + "\t" + str(round(Decimal(interval[1]),3)) + "\t" + nameAudio + "\n"
        #print(f"{interval[0]:.3f} {interval[1]:.3f}")

    file_path = file_path.replace(".csv", ".tsv")
    f = open(file_path, "w")
    f.write(output)
    f.close()

    sampleCSV = "path,sentence\n"
    for f in filenames:
      sampleCSV = sampleCSV + f + ", \n"
    sampleCSV = sampleCSV[:-1]

    folderPath = os.path.dirname(file_path)
    samplePath = folderPath + "/sample.csv"
    #print(samplePath)
    #print(sampleCSV)
    f = open(samplePath, "w")
    f.write(sampleCSV)
    f.close()
    

def leaveOnlyLastThreeColsOfTSV(inPath, outPath):
    try:
        with open(inPath, 'r', encoding='utf-8') as infile:  # Open the input file
            lines = infile.readlines()  # Read all lines
        
        processed_lines = []
        for line in lines:
            columns = line.strip().split('\t')  # Split the line into columns using tab as the delimiter
            if len(columns) >= 3:  # Ensure there are at least 3 columns to keep
                # Keep only the last three columns (columns[2], columns[3], columns[4])
                processed_lines.append(','.join(columns[2:]) + '\n')  # Rejoin and append to the list

        with open(outPath, 'w', encoding='utf-8') as outfile:  # Open the output file for writing
            outfile.writelines(processed_lines)  # Write the processed lines to the output file

        print("File processed successfully from {} to {}.".format(inPath, outPath))

    except IOError as e:
        print("An IOError occurred: {}".format(e))
    except Exception as e:
        print("An unexpected error occurred: {}".format(e))

#===========================================================================================
# Split audio where silence is at least 700ms and considered silent below -40 dBFS
#===========================================================================================

def detect_voice_intervals(wavfile, output_csv, maxWavDuration=15):

    # Find voice regions
    wav = read_audio(wavfile)
    speech_timestamps = get_speech_timestamps(
      wav,
      vadmodel,
      return_seconds=True,  # Return speech timestamps in seconds (default is samples)
    )

    # Write voice regions into a file

    output = ""

    linePrefix = "tiername\tspeakername\t"

    for t in speech_timestamps:
      #print(t['start'])
      output += linePrefix + str(t['start']) + "\t" + str(t['end']) + "\t \n"
    output = output[:-1]

    #with open("voice-regions.txt", "w") as file: file.write(output)
    with open(output_csv, "w") as file: file.write(output)

    # Make sure there aren't any regions that are longer than the memory limit
    #leaveOnlyLastThreeColsOfTSV("voice-regions.txt", output_csv)
    #fixIntervals(wavfile,  output_csv, maxWavDuration)


def prevdetect_voice_intervals(audio_path, output_csv, min_silence_len=700, silence_thresh=-40):
    # Load the audio file
    audio = AudioSegment.from_file(audio_path)

    chunks = split_on_silence(
        audio,
        min_silence_len,
        silence_thresh
    )

    # Initialize recognizer
    recognizer = sr.Recognizer()

    intervals = []

    # Process each chunk
    start_time = 0
    for i, chunk in enumerate(chunks):
        chunk_duration = len(chunk)
        if chunk_duration > 0:
            # Use recognizer to check for voice
            chunk.export("temp_chunk.wav", format="wav")
            with sr.AudioFile("temp_chunk.wav") as source:
                audio_data = recognizer.record(source)
                try:
                    # If we receive any transcript output, we can assume there's a voice
                    recognizer.recognize_google(audio_data)
                    # Append the interval where voice is detected
                    intervals.append((start_time / 1000.0, (start_time + chunk_duration) / 1000.0))
                except sr.UnknownValueError:
                    pass  # Skip chunks where no speech is recognized
            os.remove("temp_chunk.wav")

        # Update start time for the next chunk
        start_time += chunk_duration

    # Write intervals to CSV
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        #writer.writerow(['start', 'end'])
        for start, end in intervals:
            writer.writerow([start, end])

    fixIntervals(audio_path, output_csv)
    

if (premadeTSV == "NA"):
    detect_voice_intervals(outputWav, intervalsPath)
    leaveOnlyLastThreeColsOfTSV(intervalsPath, intervalsPath)
    fixIntervals(outputWav, intervalsPath)
else:
    leaveOnlyLastThreeColsOfTSV(premadeTSV, intervalsPath)
    fixIntervals(outputWav, intervalsPath)