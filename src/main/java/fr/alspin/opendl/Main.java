package fr.alspin.opendl;


import fr.alspin.opendl.math.Matrix;
import me.tongfei.progressbar.ProgressBar;
import me.tongfei.progressbar.ProgressBarBuilder;
import me.tongfei.progressbar.ProgressBarStyle;
import org.math.plot.Plot2DPanel;
import org.math.plot.PlotPanel;
import org.math.plot.plots.ScatterPlot;

import javax.swing.*;
import java.awt.*;
import java.util.Arrays;

public class Main {

    private static void progressBar() {
        long iters = 100;
        try (ProgressBar pb = new ProgressBarBuilder()
                .setInitialMax(iters)
                .setTaskName("Loop")
                .setStyle(ProgressBarStyle.COLORFUL_UNICODE_BLOCK)
                .showSpeed()
                .build()) {
            for(int i = 1;i <= iters;i ++)
            {
                Thread.sleep(100);
                pb.stepTo(i);

            }
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    private static void simpleMatrix() {
        Matrix a = Matrix.random(5, 1);
        Matrix b = Matrix.random(5, 1);
        a.print(5, 3);
        b.print(5, 3);
        Matrix c = a.mul(b);
        c.print(5, 3);

        a.pShape();

        Matrix mat = Matrix.random(10, 2);

        Plot2DPanel plot = new Plot2DPanel();
        plot.addLinePlot("mat", mat.array());

        JFrame frame = new JFrame("matrix");
        frame.setContentPane(plot);
        frame.setVisible(true);
    }

//    private static void simpleGraph() {
//        Matrix input_1 = Data.INPUT_TRAIN_SET_LINEAR, Data.OUTPUT_TRAIN_SET_LINEAR, 1);
//        Matrix input_0 = Data.INPUT_TRAIN_SET_LINEAR, Data.OUTPUT_TRAIN_SET_LINEAR, 0);
//
//        Plot2DPanel plot = new Plot2DPanel("");
//        plot.plotCanvas.addPlot(new ScatterPlot("True", Color.GREEN, 1, 5, input_1.array()));
//        plot.plotCanvas.addPlot(new ScatterPlot("True", Color.RED, 1, 5, input_0.array()));
//        plot.addLegend("South");
//
//        JFrame frame = new JFrame("matrix");
//        frame.setSize(1200, 720);
//        frame.setContentPane(plot);
//        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
//        frame.setVisible(true);
//    }

    private static void onlyTrainSetNoLayersLinear(int nIt) {
        NeuralNetwork nn = new NeuralNetwork(2, 1);

        double[][] result = nn.train(Data.INPUT_TRAIN_SET_LINEAR.transpose(), Data.OUTPUT_TRAIN_SET_LINEAR.transpose(), 0.1, nIt).transpose().array();

        JPanel panel = new JPanel();
        panel.setLayout(new BoxLayout(panel, BoxLayout.X_AXIS));

        Plot2DPanel acc = new Plot2DPanel();
        acc.addLinePlot("Accuracy", result[1]);

        Plot2DPanel log = new Plot2DPanel();
        log.addLinePlot("LogLoss", result[0]);

        panel.add(acc);
        panel.add(log);

        JFrame frame = new JFrame("matrix");
        frame.setSize(1200, 720);
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        frame.setContentPane(panel);

        acc.setFixedBounds(1, 0, 1);
        log.setFixedBounds(1, 0, Arrays.stream(result[0]).max().getAsDouble());

        frame.setVisible(true);
    }

    private static void trainAndTestSetNoLayersLinear(int nIt) {
        NeuralNetwork nn = new NeuralNetwork(2, 1);

        Matrix[] result = nn.train(
                Data.INPUT_TRAIN_SET_LINEAR.transpose(),
                Data.OUTPUT_TRAIN_SET_LINEAR.transpose(),
                Data.INPUT_TEST_SET_LINEAR.transpose(),
                Data.OUTPUT_TEST_SET_LINEAR.transpose(),
                0.1, nIt);

        double[][] trainingHistory = result[0].transpose().array();
        double[][] testHistory = result[1].transpose().array();

        JPanel mainPanel = new JPanel();
        mainPanel.setLayout(new BoxLayout(mainPanel, BoxLayout.Y_AXIS));

        JPanel trainingPanel = new JPanel();
        trainingPanel.setLayout(new BoxLayout(trainingPanel, BoxLayout.X_AXIS));

        JPanel testPanel = new JPanel();
        testPanel.setLayout(new BoxLayout(testPanel, BoxLayout.X_AXIS));

        mainPanel.add(trainingPanel);
        mainPanel.add(testPanel);

        Plot2DPanel trainingLog = new Plot2DPanel(PlotPanel.SOUTH);
        trainingLog.addLinePlot("Training", trainingHistory[0]);

        Plot2DPanel trainingAcc = new Plot2DPanel(PlotPanel.SOUTH);
        trainingAcc.addLinePlot("Training", trainingHistory[1]);

        trainingPanel.add(trainingLog);
        trainingPanel.add(trainingAcc);

        Plot2DPanel testLog = new Plot2DPanel(PlotPanel.SOUTH);
        testLog.addLinePlot("Test", testHistory[0]);

        Plot2DPanel testAcc = new Plot2DPanel(PlotPanel.SOUTH);
        testAcc.addLinePlot("Test", testHistory[1]);

        testPanel.add(testLog);
        testPanel.add(testAcc);

        JFrame frame = new JFrame("matrix");
        frame.setSize(1200, 720);
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        frame.setContentPane(mainPanel);

        trainingLog.setFixedBounds(1, 0, Arrays.stream(trainingHistory[0]).max().getAsDouble());
        trainingAcc.setFixedBounds(1, 0, 1);
        testLog.setFixedBounds(1, 0, Arrays.stream(testHistory[0]).max().getAsDouble());
        testAcc.setFixedBounds(1, 0, 1);

        frame.setVisible(true);
    }

    private static void trainAndTestSetNoLayersCircle(int nIt) {
        NeuralNetwork nn = new NeuralNetwork(2, 1);

        Matrix[] result = nn.train(
                Data.INPUT_TRAIN_SET_CIRCLE.transpose(),
                Data.OUTPUT_TRAIN_SET_CIRCLE.transpose(),
                Data.INPUT_TEST_SET_CIRCLE.transpose(),
                Data.OUTPUT_TEST_SET_CIRCLE.transpose(),
                0.1, nIt);

        double[][] trainingHistory = result[0].transpose().array();
        double[][] testHistory = result[1].transpose().array();

        JPanel mainPanel = new JPanel();
        mainPanel.setLayout(new BoxLayout(mainPanel, BoxLayout.Y_AXIS));

        JPanel trainingPanel = new JPanel();
        trainingPanel.setLayout(new BoxLayout(trainingPanel, BoxLayout.X_AXIS));

        JPanel testPanel = new JPanel();
        testPanel.setLayout(new BoxLayout(testPanel, BoxLayout.X_AXIS));

        mainPanel.add(trainingPanel);
        mainPanel.add(testPanel);

        Plot2DPanel trainingLog = new Plot2DPanel(PlotPanel.SOUTH);
        trainingLog.addLinePlot("Training", trainingHistory[0]);

        Plot2DPanel trainingAcc = new Plot2DPanel(PlotPanel.SOUTH);
        trainingAcc.addLinePlot("Training", trainingHistory[1]);

        trainingPanel.add(trainingLog);
        trainingPanel.add(trainingAcc);

        Plot2DPanel testLog = new Plot2DPanel(PlotPanel.SOUTH);
        testLog.addLinePlot("Test", testHistory[0]);

        Plot2DPanel testAcc = new Plot2DPanel(PlotPanel.SOUTH);
        testAcc.addLinePlot("Test", testHistory[1]);

        testPanel.add(testLog);
        testPanel.add(testAcc);

        JFrame frame = new JFrame("matrix");
        frame.setSize(1200, 720);
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        frame.setContentPane(mainPanel);

        trainingLog.setFixedBounds(1, 0, Arrays.stream(trainingHistory[0]).max().getAsDouble());
        trainingAcc.setFixedBounds(1, 0, 1);
        testLog.setFixedBounds(1, 0, Arrays.stream(testHistory[0]).max().getAsDouble());
        testAcc.setFixedBounds(1, 0, 1);

        frame.setVisible(true);
    }

    private static void trainAndTestSetCircle(int nIt, int[] layers) {
        NeuralNetwork nn = new NeuralNetwork(layers, 2, 1);

        Matrix[] result = nn.train(
                Data.INPUT_TRAIN_SET_CIRCLE.transpose(),
                Data.OUTPUT_TRAIN_SET_CIRCLE.transpose(),
                Data.INPUT_TEST_SET_CIRCLE.transpose(),
                Data.OUTPUT_TEST_SET_CIRCLE.transpose(),
                0.1, nIt);

        double[][] trainingHistory = result[0].transpose().array();
        double[][] testHistory = result[1].transpose().array();

        JPanel mainPanel = new JPanel();
        mainPanel.setLayout(new BoxLayout(mainPanel, BoxLayout.Y_AXIS));

        JPanel trainingPanel = new JPanel();
        trainingPanel.setLayout(new BoxLayout(trainingPanel, BoxLayout.X_AXIS));

        JPanel testPanel = new JPanel();
        testPanel.setLayout(new BoxLayout(testPanel, BoxLayout.X_AXIS));

        mainPanel.add(trainingPanel);
        mainPanel.add(testPanel);

        Plot2DPanel trainingLog = new Plot2DPanel(PlotPanel.SOUTH);
        trainingLog.addLinePlot("Training", trainingHistory[0]);

        Plot2DPanel trainingAcc = new Plot2DPanel(PlotPanel.SOUTH);
        trainingAcc.addLinePlot("Training", trainingHistory[1]);

        trainingPanel.add(trainingLog);
        trainingPanel.add(trainingAcc);

        Plot2DPanel testLog = new Plot2DPanel(PlotPanel.SOUTH);
        testLog.addLinePlot("Test", testHistory[0]);

        Plot2DPanel testAcc = new Plot2DPanel(PlotPanel.SOUTH);
        testAcc.addLinePlot("Test", testHistory[1]);

        testPanel.add(testLog);
        testPanel.add(testAcc);

        JFrame frame = new JFrame("matrix");
        frame.setSize(1200, 720);
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        frame.setContentPane(mainPanel);

        trainingLog.setFixedBounds(1, 0, Arrays.stream(trainingHistory[0]).max().getAsDouble());
        trainingAcc.setFixedBounds(1, 0, 1);
        testLog.setFixedBounds(1, 0, Arrays.stream(testHistory[0]).max().getAsDouble());
        testAcc.setFixedBounds(1, 0, 1);

        frame.setVisible(true);
    }

    private static void trainSetCircle(int nIt, int[] layers) {
        NeuralNetwork nn = new NeuralNetwork(layers, 2, 1);

        double[][] result = nn.train(
                Data.INPUT_TRAIN_SET_CIRCLE.transpose(),
                Data.OUTPUT_TRAIN_SET_CIRCLE.transpose(),
                0.1, nIt).transpose().array();

        JPanel mainPanel = new JPanel();
        mainPanel.setLayout(new BoxLayout(mainPanel, BoxLayout.Y_AXIS));

        Plot2DPanel trainingLog = new Plot2DPanel(PlotPanel.SOUTH);
        trainingLog.addLinePlot("Training", result[0]);

        Plot2DPanel trainingAcc = new Plot2DPanel(PlotPanel.SOUTH);
        trainingAcc.addLinePlot("Training", result[1]);

        mainPanel.add(trainingLog);
        mainPanel.add(trainingAcc);

        JFrame frame = new JFrame("matrix");
        frame.setSize(1200, 720);
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        frame.setContentPane(mainPanel);

        trainingLog.setFixedBounds(1, 0, Arrays.stream(result[0]).max().getAsDouble());
        trainingAcc.setFixedBounds(1, 0, 1);

        frame.setVisible(true);
    }

    public static void main(String[] args) {
//        trainAndTestSetNoLayersLinear(100000);
//        trainAndTestSetNoLayersCircle(10000);
//        trainAndTestSetCircle(3000, new int[] {32});
//        trainAndTestSetCircle(10000, new int[] {16, 16, 16});
//        trainAndTestSetCircle(1, new int[] {4, 4});
        trainSetCircle(3000, new int[] {16, 16, 16});
//        Test.matrixTest();
    }
}