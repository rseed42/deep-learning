package com.venelinpetkov.deeplearning.examples;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.MultiImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.ResNet50;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class ResNet50Transfer {
    // Logging config
    private static Logger log = LoggerFactory.getLogger(ResNet50Transfer.class);
    private static final long rseed = 42;
    private static final Random rng = new Random(rseed);
    private static final String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    private static int labelIndex = 1;
//    // This really a quirk of the implementation
//    // List<Writable> lw = recordReader.next();
//    //     lw[0] =  NDArray shaped [1,3,50,50] (1, heightm width, channels)
//    //     lw[0] =  label as integer.

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
    private static DataSetIterator loadData(String dataParentDir, int imageWidth, int imageHeight, int imageChannels, int batchSize, int labelIndex) throws IOException {
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
     * @param numClasses
     * @return
     * @throws IOException
     */
    private static ComputationGraph buildNetwork(int numClasses) throws IOException {
        ZooModel zooModel = ResNet50.builder().build();
        ComputationGraph pretrainedNet = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);
        log.info(pretrainedNet.summary());

        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(5e-5))
                .seed(rseed)
                .build();

        // Modify only the last layer to output a softmax over the 2 classes. Keep all other weights frozen
        // and update only the output layer weights.
        return new TransferLearning.GraphBuilder(pretrainedNet)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor("flatten_1")
                .removeVertexAndConnections("fc1000")
                .addLayer(
                        "predictions",
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(2048)
                                .nOut(numClasses)
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.SOFTMAX)
                                .build()
                        , "flatten_1")
                .setOutputs("predictions")
                .build();
    }

    /**
     *
     * @param args
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {
        log.info("ResNet50 Transfer Learning Dogs and Cats Example");

        Arguments cliArgs = ArgumentParser.parseCliArguments(args);
        // Prepare to load the data
        DataSetIterator trainingData = loadData(cliArgs.trainDir, cliArgs.imageWidth, cliArgs.imageHeight, cliArgs.imageChannels, cliArgs.batchSize, labelIndex);

        log.info("Data Summary");
        int numClasses = trainingData.getLabels().size();
        log.info("Number of class labels: {}", numClasses);
        ComputationGraph transferModel = buildNetwork(numClasses);

        log.info(transferModel.summary());

        // We would like to receive some info during training ...
        transferModel.setListeners(new ScoreIterationListener(10));

        // The actual training loop
        for (int i = 0; i < cliArgs.numEpochs; i++) {
            log.info("*** Starting epoch {} ***", i);
            trainingData.reset();
            while (trainingData.hasNext())
                transferModel.fit(trainingData.next());
            log.info("*** Completed epoch {} ***", i);
        }
        // Serialize the model params to disk
        ModelSerializer.writeModel(transferModel, cliArgs.parameterFilename, true);
    }
}
