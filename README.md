Inference offline

For Singularity, you can run the program using the following command. This will use Silero VAD to find the speech segments:

`./infer-asr-cpu.sh ${pathAudiofile} ${pathBlankTranscriptions} ${tempFolderRoot} ${outputFile} ${logFile}`

For example:

`./infer-asr-cpu.sh uploaded-audio/kia-orana-rehearsal.wav "NA" yourPath/yourTempFolder yourPath/yourTempFolder/thisOutput.tsv yourPath/yourTempFolder/thisLog.txt`
