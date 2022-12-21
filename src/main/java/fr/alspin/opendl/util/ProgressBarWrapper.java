package fr.alspin.opendl.util;

import me.tongfei.progressbar.ProgressBar;
import me.tongfei.progressbar.ProgressBarBuilder;
import me.tongfei.progressbar.ProgressBarStyle;

import java.util.function.BiConsumer;
import java.util.function.Consumer;

public class ProgressBarWrapper {
    static public void wrap(int nIterations, BiConsumer<ProgressBar,Integer> callback) {
        try (ProgressBar pb = new ProgressBarBuilder()
                .setInitialMax(nIterations)
                .setTaskName("Training")
                .setStyle(ProgressBarStyle.COLORFUL_UNICODE_BLOCK)
                .showSpeed()
                .setUpdateIntervalMillis(100)
                .build()) {
            callback.accept(pb, nIterations);
        }
    }
}
