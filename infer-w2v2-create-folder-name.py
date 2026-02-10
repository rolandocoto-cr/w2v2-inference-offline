#===========================================================
# Create a folder with the name of an uploaded audio file,
# and append the date in YYMMDDHHMMSS format.
# Part of the CIM ASR code.
#
# Rolando Coto. rolando.a.coto.solano@dartmouth.edu
# Last update: February 18, 2025
#===========================================================

import os
import re
import sys
from datetime import datetime

pathAudiofile = sys.argv[1]
tempFolderRoot = sys.argv[2]

def writeFolderName(file_path, tempFolderRoot):

    # Extract the filename from the file path
    base_name = os.path.basename(file_path)
    # Remove the file extension
    name, _ = os.path.splitext(base_name)
    # Remove non-alphabetic characters
    clean_name = re.sub(r'[^a-zA-Z0-9]', '', name)
    # Get the current date and time
    current_time = datetime.now()
    # Format the timestamp as 'YYYYMMDDHHMMSS'
    timestamp = current_time.strftime("%Y%m%d%H%M%S")
    # Concatenate the cleaned name with the timestamp
    result = clean_name + timestamp

    if (tempFolderRoot[len(tempFolderRoot)-1:]) != "/":
      tempFolderRoot = tempFolderRoot + "/"
    result = tempFolderRoot + result

    f = open("tempFolder.txt", "w")
    f.write(result)
    f.close()
	
writeFolderName(pathAudiofile, tempFolderRoot)
