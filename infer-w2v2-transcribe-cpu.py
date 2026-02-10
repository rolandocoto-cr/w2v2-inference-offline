#===========================================================
# Transcribe wav audio files using Wav2Vec2.
# Part of the code for CIM ASR.
#
# Rolando Coto. rolando.a.coto.solano@dartmouth.edu
# Last update: May 13, 2025
#===========================================================

import sys

asrLang = sys.argv[1]
wavName = sys.argv[2]
dataPath = sys.argv[3]
modelPath = sys.argv[4]
checkpoint = sys.argv[5]
filenameCorrectKenlmModel= sys.argv[6]
logPath = sys.argv[7]

if (dataPath[-1:] != "/"): dataPath = dataPath + "/"
csvTest = dataPath + "sample.csv"
tsvTimes = dataPath + "outputIntervals.tsv"

if (modelPath [-1:] != "/"): checkpoint = "/" + checkpoint
checkpoint = modelPath + checkpoint + "/"

print("\nasrLang: 	" + asrLang)
print("wavFile: 	" + wavName)
print("csvTest: 	" + csvTest)
print("modelPath:	" + modelPath)
print("checkpoint:	" + checkpoint)
print("lmModel:	" + filenameCorrectKenlmModel)


#============================================================================
# Load packages
#============================================================================

from datetime import datetime
currentDateAndTime = datetime.now()
startTime = str(currentDateAndTime)

import os

import transformers
import datasets
import torch

import numpy
import numpy as np

import pandas
import pandas as pd

from datasets import load_dataset, load_metric
from datasets import Dataset
from datasets import ClassLabel
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from transformers import Wav2Vec2ForCTC
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import TrainingArguments
from transformers import Trainer
from transformers import Wav2Vec2ProcessorWithLM

import pyctcdecode
from pyctcdecode import build_ctcdecoder

import random
import re
import json

import torchaudio
import librosa

from jiwer import wer
import statistics

from multiprocessing import get_context

#============================================================================
# Load CSV files and prepare dataset
#============================================================================

dataTest = pd.read_csv(csvTest)

dataTest.head()

common_voice_test = Dataset.from_pandas(dataTest)
common_voice_test_transcription = Dataset.from_pandas(dataTest)

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'

def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
    return batch
	
common_voice_test = common_voice_test.map(remove_special_characters)
	
def extract_all_chars(batch):
  all_text = " ".join(batch["sentence"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}
  
vocab_test = common_voice_test.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_test.column_names)

#============================================================================
# Load model
#============================================================================

model = Wav2Vec2ForCTC.from_pretrained(checkpoint).to("cpu")
processor = Wav2Vec2Processor.from_pretrained(modelPath)

vocab_dict = processor.tokenizer.get_vocab()
sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}

tempDecoder = build_ctcdecoder(
	labels=list(sorted_vocab_dict.keys()),
	kenlm_model_path=filenameCorrectKenlmModel,
)

processor_with_lm = Wav2Vec2ProcessorWithLM(
	feature_extractor=processor.feature_extractor,
	tokenizer=processor.tokenizer,
	decoder=tempDecoder
)


#============================================================================
# Preprocess data
#============================================================================

def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["sentence"]
    return batch

common_voice_test = common_voice_test.map(speech_file_to_array_fn, remove_columns=common_voice_test.column_names)

def resample(batch):
    batch["speech"] = librosa.resample(np.asarray(batch["speech"]), 48_000, 16_000)
    batch["sampling_rate"] = 16_000
    return batch

def prepare_dataset(batch):
    # check that all files have the correct sampling rate
    assert (
        len(set(batch["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

    batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values

    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch

common_voice_test = common_voice_test.map(prepare_dataset, remove_columns=common_voice_test.column_names, batch_size=8, num_proc=4, batched=True)


#===============================================================================
# Transcribe files
#===============================================================================

prediction = []
predictionLM = []
reference = []
paths = []

for i in range(0,len(common_voice_test)):

	input_dict = processor_with_lm(common_voice_test[i]["input_values"], sampling_rate=16_000, return_tensors="pt", padding=True)
	logits = model(input_dict.input_values.to("cpu")).logits
	pred_ids = torch.argmax(logits, dim=-1)[0]

	#print("Prediction:")
	with get_context("fork").Pool(processes=4) as pool:
		predictionlm = processor_with_lm.batch_decode(logits.cpu().detach().numpy(), pool).text[0]
	predictionLM.append(predictionlm)
	#if (i == 0): print("LM Prediction: " + str(predictionlm))
	#print("LM Prediction: " + str(predictionlm))

	path = common_voice_test_transcription[i]["path"]
	path = path.split("/")
	path = path[-1]
	paths.append(path)

#===============================================================================
# Save transcriptions
#===============================================================================

linesTSV = []

with open(tsvTimes, 'r') as file:
	# Read each line and add it to the list
	linesTSV = file.readlines()
linesTSV = [line.strip() for line in linesTSV]

outputTranscriptions = "start\tend\ttranscription\n"

for l in linesTSV:

	l = l.split("\t")
	#print(l[2])
	tsvWaveChunk = os.path.basename(l[2])

	for i in range(0,len(predictionLM)):
		#print(paths[i])
		if (tsvWaveChunk == paths[i]):
			outputTranscriptions = outputTranscriptions + l[0] + "\t" + l[1] + "\t" + predictionLM[i] + "\n"

outputTranscriptions = outputTranscriptions[:-1]

tsvOutputputFilename = dataPath + os.path.basename(wavName.replace(".wav",".tsv"))
tsvTempPath = dataPath + os.path.basename("output-filename.txt")

file = open(tsvOutputputFilename, 'w')
file.write(outputTranscriptions)
file.close()

file = open(tsvTempPath, 'w')
file.write(tsvOutputputFilename)
file.close()

file = open(logPath, 'w')
file.write("200")
file.close()


print("\nThe results are stored in: \n" + tsvOutputputFilename)