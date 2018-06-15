package com.venelinpetkov.deeplearning.examples;

import org.datavec.image.loader.BaseImageLoader;
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
import java.io.IOException;
import java.util.Random;

public class ResNet50Transfer {
    // Logging config
    private static Logger log = LoggerFactory.getLogger(ResNet50Transfer.class);
    private static final String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    private static int labelIndex = 1;

    /**
     *
     * @param numClasses
     * @return
     * @throws IOException
     */
    private static ComputationGraph buildNetwork(int numClasses, int rseed) throws IOException {
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

        // We need this for some other stuff actually ...
        final Random rng = new Random(cliArgs.randomSeed);

        // Prepare to load the data
        DataSetIterator trainingData = DataLoaders.loadData(
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
        int numClasses = trainingData.getLabels().size();
        log.info("Number of class labels: {}", numClasses);
        ComputationGraph transferModel = buildNetwork(numClasses, cliArgs.randomSeed);

        if (cliArgs.showArchitecture)
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
