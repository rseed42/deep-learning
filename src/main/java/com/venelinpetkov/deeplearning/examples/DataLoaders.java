package com.venelinpetkov.deeplearning.examples;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.MultiImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class DataLoaders {
    /**
     *
     * @param dataParentDir
     * @param imageWidth
     * @param imageHeight
     * @param imageChannels
     * @param batchSize
     * @param labelIndex
     * @return
     * @throws IOException
     */
    public static DataSetIterator loadDataSplit(
            String dataParentDir,
            int imageWidth,
            int imageHeight,
            int imageChannels,
            int batchSize,
            int labelIndex,
            final String[] allowedExtensions,
            Random rng
    ) throws IOException {
        // We do not have a separate train / test data set, so we can use
        // a file split to do this automatically for us. All of the data is
        // located in the parent directory:
        File parentDir = new File(dataParentDir);
        // Specify what files we are to use, as well as an rng for the split:
        FileSplit fileSplit = new FileSplit(parentDir, allowedExtensions, rng);
        // We use the name of each class subdirectory as the class name, so we have
        // to use the corresponding label generator:
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        // Randomizes the order of paths in an array and removes paths randomly to have the same number of paths
        // for each label. Further interlaces the paths on output based on their labels,
        // to obtain easily optimal batches for training.
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, allowedExtensions, labelMaker);
        // Specify how to split the data into training / testing data sets with 80%/20% ratio:
        InputSplit[] filesInDirSplit = fileSplit.sample(pathFilter, 80, 20);
        InputSplit trainData = filesInDirSplit[0];
        InputSplit testData = filesInDirSplit[1];
        // The ImageRecordReader will automatically scale the images to the given dimensions
        ImageRecordReader recordReader = new ImageRecordReader(imageHeight, imageWidth, imageChannels, labelMaker);
        // We can transform the images for data augmentation purposes
        ImageTransform transform = new MultiImageTransform(rng);

        // Initialize the record reader
        recordReader.initialize(trainData, transform);
        // Finally, we can create the data set
        return new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, recordReader.numLabels());
    }

    /**
     *
     * @param dataParentDir
     * @param imageWidth
     * @param imageHeight
     * @param imageChannels
     * @param batchSize
     * @param labelIndex
     * @return
     * @throws IOException
     */
    public static DataSetIterator loadDataFull(
            String dataParentDir,
            int imageWidth,
            int imageHeight,
            int imageChannels,
            int batchSize,
            int labelIndex,
            final String[] allowedExtensions,
            Random rng
    ) throws IOException {
        // We do not have a separate train / test data set, so we can use
        // a file split to do this automatically for us. All of the data is
        // located in the parent directory:
        File parentDir = new File(dataParentDir);
        // Specify what files we are to use, we do not split the data set:
        FileSplit fileSplit = new FileSplit(parentDir, allowedExtensions, rng);
        // We use the name of each class subdirectory as the class name, so we have
        // to use the corresponding label generator:
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        // The ImageRecordReader will automatically scale the images to the given dimensions
        ImageRecordReader recordReader = new ImageRecordReader(imageHeight, imageWidth, imageChannels, labelMaker);
        // We can transform the images for data augmentation purposes
        ImageTransform transform = new MultiImageTransform(rng);
        // Initialize the record reader
        recordReader.initialize(fileSplit, transform);
        return new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, recordReader.numLabels());
    }
}
