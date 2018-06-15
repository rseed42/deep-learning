package com.venelinpetkov.deeplearning.examples;

import org.datavec.image.loader.BaseImageLoader;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
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
        DataSetIterator testingData = DataLoaders.loadDataFull(
                cliArgs.trainDir,
                cliArgs.imageWidth,
                cliArgs.imageHeight,
                cliArgs.imageChannels,
                cliArgs.batchSize,
                labelIndex,
                allowedExtensions,
                rng
        );

        log.info("Data Summary");
        int numClasses = testingData.getLabels().size();
        log.info("Number of class labels: {}", numClasses);

        // Load the model from file
        Nd4j.getRandom().setSeed(cliArgs.randomSeed);
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(cliArgs.parameterFilename);
        // Evaluate (distributed)
        Evaluation evaluation = new Evaluation(numClasses);
        model.doEvaluation(testingData, evaluation);
        log.info("*** Evaluation ***");
        log.info(evaluation.stats());
    }
}
