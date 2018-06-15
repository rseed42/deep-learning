package com.venelinpetkov.deeplearning.examples;

import org.datavec.image.loader.BaseImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Random;

public class ResNet50TransferEvaluate {
    // Logging config
    private static Logger log = LoggerFactory.getLogger(ResNet50TransferEvaluate.class);
    private static final String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    private static final int labelIndex = 1;

    public static void main(String[] args) throws IOException {
        log.info("ResNet50 Transfer Learning Evaluation Dogs and Cats Example");

        Arguments cliArgs = ArgumentParser.parseCliArguments(args);

        // We need this for some other stuff actually ...
        final Random rng = new Random(cliArgs.randomSeed);

        // Prepare to load the data
        DataSetIterator trainingData = DataLoaders.loadDataFull(
                cliArgs.trainDir,
                cliArgs.imageWidth,
                cliArgs.imageHeight,
                cliArgs.imageChannels,
                cliArgs.batchSize,
                labelIndex,
                allowedExtensions,
                rng
        );
//
//        log.info("Data Summary");
//        int numClasses = trainingData.getLabels().size();
//        log.info("Number of class labels: {}", numClasses);
//        ComputationGraph transferModel = ResNet50TransferNetFactory.buildNetwork(numClasses, cliArgs.randomSeed);
//
//        if (cliArgs.showArchitecture)
//            log.info(transferModel.summary());
//
//        // We would like to receive some info during training ...
//        transferModel.setListeners(new ScoreIterationListener(cliArgs.scorePerIteration));
//
//        // The actual training loop
//        for (int i = 0; i < cliArgs.numEpochs; i++) {
//            log.info("*** Starting epoch {} ***", i);
//            trainingData.reset();
//            while (trainingData.hasNext())
//                transferModel.fit(trainingData.next());
//            log.info("*** Completed epoch {} ***", i);
//        }
//
//        // Serialize the model params to disk
//        ModelSerializer.writeModel(transferModel, cliArgs.parameterFilename, true);
    }
}
