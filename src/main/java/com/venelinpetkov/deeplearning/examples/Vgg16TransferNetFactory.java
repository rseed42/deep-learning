package com.venelinpetkov.deeplearning.examples;

import org.datavec.image.loader.BaseImageLoader;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class Vgg16TransferNetFactory {

    // Logging config
    private static Logger log = LoggerFactory.getLogger(Vgg16TransferNetFactory.class);

    /**
     *
     * @param numClasses
     * @return
     * @throws IOException
     */
    public static ComputationGraph buildNetwork(int numClasses, int rseed) throws IOException {
        ZooModel zooModel = VGG16.builder().build();
        ComputationGraph pretrainedNet = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);

        log.info(pretrainedNet.summary());

        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(5e-5))
                .seed(rseed)
                .build();

        // Modify only the last layer to output a softmax over the 5 classes. Keep all other weights frozen
        // and update only the output layer weights.
        return new TransferLearning.GraphBuilder(pretrainedNet)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor("fc2")
                .removeVertexKeepConnections("predictions")
                .addLayer(
                        "predictions",
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(4096)
                                .nOut(numClasses)
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.SOFTMAX)
                                .build()
                        , "fc2")
                .build();
    }
}
