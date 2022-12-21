package fr.alspin.opendl;

import fr.alspin.opendl.math.Matrix;
import fr.alspin.opendl.math.Function;
import fr.alspin.opendl.util.ProgressBarWrapper;
import me.tongfei.progressbar.ProgressBar;
import me.tongfei.progressbar.ProgressBarBuilder;
import me.tongfei.progressbar.ProgressBarStyle;
import org.jetbrains.annotations.NotNull;

import java.util.*;
import java.util.function.Consumer;

public class NeuralNetwork {

    private int nNeuron;
    private Matrix[][] parameters;
    private Matrix[] activations;
    private Matrix[][] gradients;

    public NeuralNetwork(int[] layers, int nInput, int nOutput) {
        List<Integer> dimensions = new ArrayList<>();
        dimensions.add(nInput);
        dimensions.addAll(Arrays.stream(layers).boxed().toList());
        dimensions.add(nOutput);

        this.initialisation(dimensions);
    }

    public NeuralNetwork(int nInput, int nOutput) {
        List<Integer> dimensions = new ArrayList<>();
        dimensions.add(nInput);
        dimensions.add(nOutput);

        this.initialisation(dimensions);
    }

    private void initialisation(List<Integer> dims) {
        this.parameters = new Matrix[2][dims.size()];
        this.gradients = new Matrix[2][dims.size()];

        this.nNeuron = Math.floorDiv(2 * (dims.size() - 1), 2);
        this.activations = new Matrix[this.nNeuron + 1];

        for (int i = 1; i < dims.size(); i++) {
            this.parameters[0][i] = Matrix.gaussian(dims.get(i), dims.get(i - 1));
            this.parameters[1][i] = Matrix.gaussian(dims.get(i), 1);
        }
    }

    private void forwardPropagation(Matrix input) {
        this.activations[0] = input;

        for (int i = 1; i <= this.nNeuron ; i++) {
            Matrix Z = this.parameters[0][i].dot(this.activations[i - 1]).add(this.parameters[1][i]);
            this.activations[i] = Function.sigmoid(Z);
        }
    }

    private void backPropagation(@NotNull Matrix output) {
        double m = output.n();

        Matrix dZ = this.activations[this.nNeuron].sub(output);

        for (int i = this.nNeuron; i > 0; i--) {
            this.gradients[0][i] = dZ.dot(this.activations[i - 1].transpose()).mul(1.0 / m);
            this.gradients[1][i] = dZ.sum(Matrix.Axis.ROW).mul(1.0 / m);

            if(i > 1) {
                dZ = this.parameters[0][i]
                        .transpose()
                        .dot(dZ)
                        .mul(this.activations[i - 1])
                        .mul(this.activations[i - 1].opposite().add(1));
            }
        }
    }

    private void update(double learningRate) {
        for (int i = 1; i <= this.nNeuron; i++) {
            this.parameters[0][i] = this.parameters[0][i].sub(this.gradients[0][i].mul(learningRate));
            this.parameters[1][i] = this.parameters[1][i].sub(this.gradients[1][i].mul(learningRate));
        }
    }

    public Matrix predict(Matrix input) {
        forwardPropagation(input);
        return this.activations[this.nNeuron].compare((val) -> val >= 0.5 ? 1. : 0.);
    }

    public Matrix train(Matrix inputTrainSet, Matrix outputTrainSet, double learningRate, int nIteration) {
        nIteration = Math.max(nIteration, 1000);
        Matrix trainingHistory = new Matrix(1000, 2, 0);

        ProgressBarWrapper.wrap(nIteration, (pb, iter) -> {
            for (int i = 1; i <= iter; i++) {

                forwardPropagation(inputTrainSet);
                backPropagation(outputTrainSet);
                update(learningRate);

                if(i % (iter / 1000) == 0) {
                    trainingHistory.set((i - 1) / (iter / 1000), 0, Function.logLoss(outputTrainSet.flatten(), this.activations[this.nNeuron].flatten()));
                    trainingHistory.set((i - 1) / (iter / 1000), 1, Function.accuracyScore(outputTrainSet.flatten(), predict(inputTrainSet).flatten()));
                }

                pb.stepTo(i);
            }
        });
        return trainingHistory;
    }
}
