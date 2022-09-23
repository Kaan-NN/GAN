from random import randint, random
import numpy as np
from PIL import Image as img
import numpy as np
from pathlib import Path

def preprocess(pathToData, pathToProcessedData = None):
    if(not Path(pathToData).exists):
        raise OSError(pathToData)
    for file in Path(pathToData).iterdir():
        print(f"{file.name} has been converted to a useable format")
        if file.suffix in ['.png', '.jpeg', '.jpg']:
            prepareTrainingData(file, pathToProcessedData)
    print("##########Image preprocessing complete##########")

def prepareTrainingData(rawFile, pathToProcessedData = None):
    with img.open(rawFile) as rawImage:
        preProcessedFile = Path(pathToProcessedData).joinpath(Path(rawFile.stem).with_suffix(".npy"))
        print(preProcessedFile)
        tensor = np.array(rawImage)
        np.save(preProcessedFile, tensor)

def randomDropoutMatrix(dropoutRate, matrixShape):
    return np.random.binomial(np.ones(matrixShape, dtype=int), 1-dropoutRate)