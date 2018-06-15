package com.venelinpetkov.deeplearning.examples;
import com.beust.jcommander.Parameter;

public class Arguments {

    @Parameter(names = "-batchLen", description = "Minibatch size")
    public int batchSize = 10;

    @Parameter(names = "-epochs", description = "Number of epochs for training")
    public int numEpochs = 1;
}

