package fr.alspin.opendl.math;

import org.jetbrains.annotations.NotNull;
import java.util.Arrays;

public class Function {

    public static double sigmoid(double value) {
        return (1 / (1 + Math.exp(-value)));
    }

    @NotNull
    public static Matrix sigmoid(@NotNull Matrix matrix) {
        Matrix result = new Matrix(matrix.shape(), 0);
        for (int row = 0; row < matrix.m(); row++) {
            for (int col = 0; col < matrix.n(); col++) {
                result.set(row, col, sigmoid(matrix.get(row, col)));
            }
        }
        return result;
    }

    @NotNull
    public static Matrix log(@NotNull Matrix matrix) {
        Matrix result = Matrix.zero(matrix.shape());
        matrix.foreach((i, j, val) -> result.set(i, j, Math.log(val)));
        return result;
    }

    public static double logLoss(@NotNull Matrix a, @NotNull Matrix b) {
        if(!Arrays.equals(a.shape(), b.shape())) throw new RuntimeException("Matrix must be flatten to be use by logLoss method.");
        double epsilon = Math.exp(-15);
        return 1.0 / (b.n()) * b.opposite().mul(log(a.add(epsilon))).sub(b.opposite().add(1).mul(log(a.opposite().add(epsilon).add(1)))).sum();
    }

    public static double accuracyScore(@NotNull Matrix a, @NotNull Matrix b) {
        if(!Arrays.equals(a.shape(), b.shape())) throw new RuntimeException("Matrix must be flatten to be use by accuracyScore method.");
        double result = 0;
        for (int i = 0; i < a.n(); i++) {
            result += Math.abs(a.get(0, i) - b.get(0, i));
        }
        result = 1 - result / a.n();
        return result;
    }
}
