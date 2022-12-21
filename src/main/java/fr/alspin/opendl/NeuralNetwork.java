package fr.alspin.opendl;

import fr.alspin.opendl.math.Matrix;
import fr.alspin.opendl.math.Function;
import me.tongfei.progressbar.ProgressBar;
import me.tongfei.progressbar.ProgressBarBuilder;
import me.tongfei.progressbar.ProgressBarStyle;
import org.jetbrains.annotations.NotNull;

import java.util.*;
import java.util.function.Consumer;

public class NeuralNetwork {

    private final List<Integer> dimensions = new ArrayList<>();
    private final Map<String, Matrix> parameters = new HashMap<>();
    private final Map<String, Matrix> activations = new HashMap<>();
    private final Map<String, Matrix> gradients = new HashMap<>();

    private int nNeuron;

    public NeuralNetwork(int[] layers, int nInput, int nOutput) {
        this.dimensions.add(nInput);
        this.dimensions.addAll(Arrays.stream(layers).boxed().toList());
        this.dimensions.add(nOutput);

        this.initialisation();
    }

    public NeuralNetwork(int nInput, int nOutput) {
        this.dimensions.add(nInput);
        this.dimensions.add(nOutput);

        this.initialisation();
    }

    private void initialisation() {
        for (int i = 1; i < this.dimensions.size(); i++) {
            this.parameters.put("W" + i, Matrix.gaussian(this.dimensions.get(i), this.dimensions.get(i - 1)));
            this.parameters.put("b" + i, Matrix.gaussian(this.dimensions.get(i), 1));
        }
        this.nNeuron = Math.floorDiv(this.parameters.size(), 2);
    }

    private void forwardPropagation(Matrix input) {
        if(this.activations.putIfAbsent("A0", input) != null)
            this.activations.replace("A0", input);

        for (int i = 1; i <= this.nNeuron ; i++) {
            Matrix Z = this.parameters.get("W" + i).dot(this.activations.get("A" + (i - 1))).add(this.parameters.get("b" + i));

            if(this.activations.putIfAbsent("A" + i, Function.sigmoid(Z)) != null)
                this.activations.replace("A" + i, Function.sigmoid(Z));
        }
    }

    private void backPropagation(@NotNull Matrix output) {
        double m = output.shape()[1];
        Matrix dZ = this.activations.get("A" + nNeuron).sub(output);

        for (int i = this.nNeuron; i > 0; i--) {
            if(this.gradients.putIfAbsent("dW" + i, dZ.dot(this.activations.get("A" + (i - 1)).transpose()).mul(1.0 / m)) != null) {
                this.gradients.replace("dW" + i, dZ.dot(this.activations.get("A" + (i - 1)).transpose()).mul(1.0 / m));
            }
            if(this.gradients.putIfAbsent("db" + i, dZ.sum(Matrix.Axis.ROW).mul(1.0 / m)) != null) {
                this.gradients.replace("db" + i, dZ.sum(Matrix.Axis.ROW).mul(1.0 / m));
            }
            if(i > 1) {
                dZ = this.parameters.get("W" + i)
                        .transpose()
                        .dot(dZ)
                        .mul(this.activations.get("A" + (i - 1))
                        .mul(this.activations.get("A" + (i - 1)).opposite().add(1)));
            }
        }
    }

    private void update(double learningRate) {
        for (int i = 1; i <= this.nNeuron; i++) {
            this.parameters.replace("W" + i, this.parameters.get("W" + i).sub(this.gradients.get("dW" + i).mul(learningRate)));
            this.parameters.replace("b" + i, this.parameters.get("b" + i).sub(this.gradients.get("db" + i).mul(learningRate)));
        }
    }

    public Matrix predict(Matrix input) {
        forwardPropagation(input);
        return this.activations.get("A" + this.nNeuron).compare((val) -> val >= 0.5 ? 1. : 0.);
    }

    private Matrix[] predictAndActivation(Matrix input) {
        return new Matrix[] {predict(input), this.activations.get("A" + this.nNeuron)};
    }

    public Matrix train(Matrix inputTrainSet, Matrix outputTrainSet, double learningRate, int nIteration, Consumer<Integer> consumer) {
        Matrix trainingHistory = new Matrix(Math.min(nIteration, 1000), 2, 0);
        Matrix outputPrediction;
        try (ProgressBar pb = new ProgressBarBuilder()
                .setInitialMax(nIteration)
                .setTaskName("Loop")
                .setStyle(ProgressBarStyle.COLORFUL_UNICODE_BLOCK)
                .showSpeed()
                .build()) {
            for (int i = 1; i <= nIteration; i++) {
                forwardPropagation(inputTrainSet);
                backPropagation(outputTrainSet);
                update(learningRate);

                if(nIteration > 1000) {
                    if(i % (nIteration / 1000) == 0) {
                        trainingHistory.set((i - 1) / (nIteration / 1000), 0, Function.logLoss(outputTrainSet.flatten(), activations.get("A" + this.nNeuron).flatten()));
                        outputPrediction = predict(inputTrainSet);
                        trainingHistory.set((i - 1) / (nIteration / 1000), 1, Function.accuracyScore(outputTrainSet.flatten(), outputPrediction.flatten()));
                        if(consumer != null)
                            consumer.accept(i);
                    }
                } else {
                    trainingHistory.set(i - 1, 0, Function.logLoss(outputTrainSet.flatten(), activations.get("A" + this.nNeuron).flatten()));
                    outputPrediction = predict(inputTrainSet);
                    trainingHistory.set(i - 1, 1, Function.accuracyScore(outputTrainSet.flatten(), outputPrediction.flatten()));
                    if(consumer != null)
                        consumer.accept(i);
                }
                if(consumer != null)
                    consumer.accept(i);

                pb.stepTo(i);
            }
        }
        return trainingHistory;
    }

    public Matrix train(Matrix inputTrainSet, Matrix outputTrainSet, double learningRate, int nIteration) {
        return train(inputTrainSet, outputTrainSet, learningRate, nIteration, null);
    }

    public Matrix[] train(Matrix inputTrainSet, Matrix outputTrainSet, Matrix inputTestSet, Matrix outputTestSet, double learningRate, int nIteration) {
        Matrix testingHistory = new Matrix(Math.min(nIteration, 1000), 2, 0);
        Matrix trainingHistory = train(inputTrainSet, outputTrainSet, learningRate, nIteration, (i) -> {
            Matrix[] outputPrediction;
            if(nIteration > 1000) {
                if(i % (nIteration / 1000) == 0) {
                    outputPrediction = predictAndActivation(inputTestSet);
                    testingHistory.set((i - 1) / (nIteration / 1000), 0, Function.logLoss(outputTestSet.flatten(), outputPrediction[1].flatten()));
                    testingHistory.set((i - 1) / (nIteration / 1000), 1, Function.accuracyScore(outputTestSet.flatten(), outputPrediction[0].flatten()));
                }
            } else {
                outputPrediction = predictAndActivation(inputTestSet);
                testingHistory.set(i - 1, 0, Function.logLoss(outputTestSet.flatten(), outputPrediction[1].flatten()));
                testingHistory.set(i - 1, 1, Function.accuracyScore(outputTestSet.flatten(), outputPrediction[0].flatten()));
            }
        });
        return new Matrix[] {trainingHistory, testingHistory};
    }
}
