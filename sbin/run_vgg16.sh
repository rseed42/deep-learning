#!/bin/bash
JAR_DIR=./build/libs
CLASS_NAME=com.venelinpetkov.deeplearning.examples.Vgg16TransferLearn

# Run the command
java -cp $JAR_DIR/*.jar $CLASS_NAME \
     -batchSize 64 \
     -numEpochs 1 \
     -trainDir "/data/dogscats/train" \
     -parameterFilename "vgg16.zip" \
     -showArchitecture false
