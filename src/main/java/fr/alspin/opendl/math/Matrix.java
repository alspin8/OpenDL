package fr.alspin.opendl.math;

import fr.alspin.opendl.util.TriConsumer;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;

import java.io.Serializable;
import java.security.InvalidParameterException;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.Function;

public class Matrix implements Serializable {

    public enum Axis { ROW, COLUMN }

    private int m;
    private int n;
    private double[][] data;

    public Matrix(@NotNull double [][] data) {
        this.m = data.length;
        this.n = data[0].length;
        this.data = data;
    }

    public Matrix(int m, int n, double value) {
        double[][] data = new double[m][n];
        Arrays.stream(data).forEach(a -> Arrays.fill(a, value));
        this.m = m;
        this.n = n;
        this.data = data;
    }

    public Matrix(int[] shape, double value) {
        this(shape[0], shape[1], value);
    }

    private Matrix(int m, int n, double[][] data) {
        this.m = m;
        this.n = n;
        this.data = data;
    }

    private Matrix(int size, double[] data, Axis axis) {
        double[][] d = new double[axis != Axis.ROW ? data.length : size][axis != Axis.COLUMN ? data.length : size];
        if(axis == Axis.ROW) {
            for (int i = 0; i < size; i++) {
                d[i] = data;
            }
        } else if (axis == Axis.COLUMN) {
            for (int i = 0; i < data.length; i++) {
                for (int j = 0; j < size; j++) {
                    d[i][j] = data[i];
                }
            }
        }
        this.m = d.length;
        this.n = d[0].length;
        this.data = d;
    }

    @NotNull
    @Contract("_, _ -> new")
    public static Matrix zero(int m, int n) {
        return new Matrix(m, n, 0);
    }

    @NotNull
    @Contract("_ -> new")
    public static Matrix zero(int[] shape) {
        return new Matrix(shape, 0);
    }

    @NotNull
    @Contract("_, _, _ -> new")
    public static Matrix random(int m, int n, int... bound) {
        bound = bound.length == 2 && bound[0] < bound[1] ? bound : new int[] {0, 1};
        int min = bound[0];
        int max = bound[1];
        double[][] data = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                data[i][j] = (Math.random() * (max - min)) + min;
            }
        }
        return new Matrix(m, n, data);
    }

    @NotNull
    @Contract("_, _ -> new")
    public static Matrix random(int[] shape, int... bound) {
        return random(shape[0], shape[1], bound);
    }

    @NotNull
    @Contract("_, _, _, _ -> new")
    public static Matrix gaussian(int m, int n, double mean, double variance) {
        Random rnd = new Random();
        double[][] data = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                data[i][j] = mean + rnd.nextGaussian() * variance;
            }
        }
        return new Matrix(m, n, data);
    }

    @NotNull
    @Contract("_, _ -> new")
    public static Matrix gaussian(int m, int n) {
        return gaussian(m, n, 0, 1);
    }

    public int m() {
        return m;
    }

    public int n() {
        return n;
    }

    private static boolean checkDimension(Matrix a, Matrix b, String mode) {
        return switch (mode) {
            case "same" -> Arrays.equals(a.shape(), b.shape());
            case "mul" -> a.n == b.m;
            default -> false;
        };
    }

    public void foreach(TriConsumer<Integer, Integer, Double> callback) {
        for (int i = 0; i < this.m; i++) {
            for (int j = 0; j < this.n; j++) {
                callback.accept(i, j, this.data[i][j]);
            }
        }
    }

    public void foreach(BiConsumer<Integer, Integer> callback) {
        for (int i = 0; i < this.m; i++) {
            for (int j = 0; j < this.n; j++) {
                callback.accept(i, j);
            }
        }
    }

    public void foreach(BiFunction<Integer, Integer, Double> callback) {
        for (int i = 0; i < this.m; i++) {
            for (int j = 0; j < this.n; j++) {
                this.data[i][j] = callback.apply(i, j);
            }
        }
    }

    public void set(int m, int n, double value) {
        this.data[m][n] = value;
    }

    public double get(int m, int n) {
        return this.data[m][n];
    }

    private void add(int m, int n, double value) {
        this.data[m][n] += value;
    }

    public Matrix add(double value) {
        return this.add(new Matrix(this.m, this.n, value), false);
    }

    public Matrix add(double value, boolean keep) {
        return this.add(new Matrix(this.m, this.n, value), keep);
    }

    public Matrix add(Matrix m) {
        return this.add(m, false);
    }

    public Matrix add(Matrix m, boolean keep) {
        boolean sameDims = checkDimension(this, m, "same");

        if(m.m == 1 && m.n == 1 && !sameDims) {
            return this.add(m.get(0, 0), keep);
        } else if (m.m == 1 && m.n == this.n && !sameDims) {
            return this.add(new Matrix(this.m, m.flatten().array()[0], Axis.ROW), keep);
        } else if (m.n == 1 && m.m == this.m && !sameDims) {
            return this.add(new Matrix(this.n, m.flatten().array()[0], Axis.COLUMN), keep);
        } else if (!sameDims) {
            throw new RuntimeException("Matrices must have same size to be added.");
        }

        if(keep) {
            this.foreach((i, j) -> this.data[i][j] + m.data[i][j]);
            return this;
        } else {
            double[][] data = new double[this.m][this.n];
            this.foreach((BiConsumer<Integer, Integer>) (i, j) -> data[i][j] = this.data[i][j] + m.data[i][j]);
            return new Matrix(data);
        }
    }

    private void sub(int m, int n, double value) {
        this.data[m][n] -= value;
    }

    public Matrix sub(double value) {
        return this.sub(new Matrix(this.m, this.n, value), false);
    }

    public Matrix sub(double value, boolean keep) {
        return this.sub(new Matrix(this.m, this.n, value), keep);
    }

    public Matrix sub(Matrix m) {
        return this.sub(m, false);
    }

    public Matrix sub(Matrix m, boolean keep) {
        if(!checkDimension(this, m, "same"))
            throw new RuntimeException("Matrices must have same size to be subtracted.");
        if(keep) {
            this.foreach((i, j) -> this.data[i][j] - m.data[i][j]);
            return this;
        } else {
            double[][] data = new double[this.m][this.n];
            this.foreach((BiConsumer<Integer, Integer>) (i, j) -> data[i][j] = this.data[i][j] - m.data[i][j]);
            return new Matrix(data);
        }
    }

    public Matrix mul(double value) {
        return this.mul(value, false);
    }

    public Matrix mul(double value, boolean keep) {
        return this.mul(new Matrix(this.m, this.n, value), keep);
    }

    public Matrix mul(Matrix m) {
        return this.mul(m, false);
    }

    public Matrix mul(Matrix m, boolean keep) {
        if(!checkDimension(this, m, "same"))
            throw new RuntimeException("Matrices must have same size to be multiply.");
        if(keep) {
            this.foreach((i, j) -> this.data[i][j] * m.data[i][j]);
            return this;
        } else {
            double[][] data = new double[this.m][this.n];
            this.foreach((BiConsumer<Integer, Integer>) (i, j) -> data[i][j] = this.data[i][j] * m.data[i][j]);
            return new Matrix(data);
        }
    }

    public Matrix dot(Matrix m) {
        if(!checkDimension(this, m, "mul"))
            throw new RuntimeException("Matrix A must have N equals to M of matrix B.");
        Matrix result = Matrix.zero(this.m, m.n);
        double[] row;
        for (int i = 0; i < this.m; i++) {
            row = data[i];
            for (int j = 0; j < m.n; j++) {
                for (int k = 0; k < row.length; k++) {
                    result.add(i, j, row[k] * m.data[k][j]);
                }
            }
        }
        return result;
    }

    public Matrix transpose() {
        return this.transpose(false);
    }

    public Matrix transpose(boolean keep) {
        double[][] data = new double[this.n][this.m];
        this.foreach((BiConsumer<Integer, Integer>) (i, j) -> data[j][i] = this.data[i][j]);
        if(keep) {
            this.n = this.m;
            this.m = data.length;
            this.data = data;
            return this;
        } else {
            return new Matrix(data);
        }
    }

    public Matrix opposite() {
        Matrix result = Matrix.zero(this.shape());
        foreach((i, j, val) ->  result.set(i, j, -val));
        return result;
    }

    public Matrix compare(Function<Double, Double> callback) {
        Matrix result = Matrix.zero(this.shape());
        for (int i = 0; i < this.m; i++) {
            for (int j = 0; j < this.n; j++) {
                result.set(i, j, callback.apply(this.get(i, j)));
            }
        }
        return result;
    }

    public Matrix flatten() {
        Matrix result = Matrix.zero(1, this.m * this.n);
        foreach((i, j, val) -> result.set(0, (i * this.n + j), val));
        return result;
    }

    public Matrix sum(Axis axis) {
        Matrix result;
        if(axis == Axis.ROW) {
            result = Matrix.zero(this.m, 1);
            foreach((i, j, val) -> result.add(i, 0, val));
        } else if (axis == Axis.COLUMN) {
            result = Matrix.zero(1, this.n);
            foreach((i, j, val) -> result.add(0, j, val));
        } else
            throw new InvalidParameterException("Unknown axis");
        return result;
    }

    public double sum() {
        AtomicReference<Double> result = new AtomicReference<>((double) 0);
        if(this.m == 1 || this.n == 1) {
            foreach((i, j, val) -> result.updateAndGet(v -> v + val));
            return result.get();
        } else throw new RuntimeException("Can't sum a matrix who hasn't at least one of its dimension equals to 1");
    }

    public static double sum(Matrix m) {
        return m.sum();
    }

    public int[] shape() {
        return new int[] {m, n};
    }

    public double[][] array() {
        return this.data;
    }

    public void pShape() {
        System.out.println("m : " + this.m + ", n : " + this.n);
    }

    public void print() {
        this.print(2, 5);
    }

    public void print(int precision, int space) {
        for (int i = 0; i < this.m; i++) {
            for (int j = 0; j < this.n; j++) {
                System.out.printf("%." + precision + "f", this.data[i][j]);
                for (int k = 0; k < space; k++) {
                    System.out.print(" ");
                }
            }
            System.out.println();
        }
    }
}
