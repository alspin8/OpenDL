package fr.alspin.opendl;

import fr.alspin.opendl.math.Function;
import fr.alspin.opendl.math.Matrix;
import org.math.plot.Plot2DPanel;
import org.math.plot.PlotPanel;
import javax.swing.*;
import java.util.Arrays;

public class Main {
    private static void trainSetCircle(int nIt, int[] layers) {
        NeuralNetwork nn = new NeuralNetwork(layers, 2, 1);

        double[][] result = nn.train(
                Data.INPUT_TRAIN_SET_CIRCLE.transpose(),
                Data.OUTPUT_TRAIN_SET_CIRCLE.transpose(),
                0.1, nIt).transpose().array();

        Matrix r2 = nn.predict(Data.INPUT_TEST_SET_CIRCLE.transpose());

        double r3 = Function.accuracyScore(Data.OUTPUT_TEST_SET_CIRCLE.transpose().flatten(), r2.flatten());
        System.out.println(r3);

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

        if(Arrays.stream(result[0]).max().isPresent())
            trainingLog.setFixedBounds(1, 0, Arrays.stream(result[0]).max().getAsDouble());
        trainingAcc.setFixedBounds(1, 0, 1);

        frame.setVisible(true);
    }

    public static void main(String[] args) {
//        trainSetCircle(1000, new int[] {16, 16, 16});
        trainSetCircle(10000, new int[] {16});
    }
}