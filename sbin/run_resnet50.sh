#!/bin/bash
JAR_DIR=../build/libs
CLASS_NAME=com.venelinpetkov.deeplearning.examples.ResNet50Transfer

# Run the command
java -cp $JAR_DIR/*.jar $CLASS_NAME \
     -batchSize 10 \
     -numEpochs 1 \
     -trainDir "/data/dogscats/train" \
     -parameterFilename "resnet50.zip"
