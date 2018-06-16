package com.venelinpetkov.deeplearning.examples;

import org.datavec.image.loader.BaseImageLoader;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Random;

public class Vgg16TransferEvaluate {
    // Logging config
    private static Logger log = LoggerFactory.getLogger(Vgg16TransferEvaluate.class);
    private static final String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    private static final int labelIndex = 1;

    public static void main(String[] args) throws IOException {
        log.info("Vgg16 Transfer Learning Evaluation Dogs and Cats Example");

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
        ComputationGraph model = ModelSerializer.restoreComputationGraph(cliArgs.parameterFilename);
        //
        // We would like to receive some info during training ...
        model.setListeners(new ScoreIterationListener(cliArgs.scorePerIteration));
        // Evaluate
        Evaluation evaluation = new Evaluation(numClasses);
        model.doEvaluation(testingData, evaluation);
        log.info("*** Evaluation ***");
        log.info(evaluation.stats());
    }
}
