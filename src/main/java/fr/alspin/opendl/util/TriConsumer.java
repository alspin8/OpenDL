package fr.alspin.opendl.util;

import java.util.Objects;

@FunctionalInterface
public interface TriConsumer<T, U, V> {
    public void accept(T a, U b, V c);

    public default TriConsumer<T, U, V> andThen(TriConsumer<? super T, ? super U, ? super V> after) {
        Objects.requireNonNull(after);
        return (a, b, c) -> {
            accept(a, b, c);
            after.accept(a, b, c);
        };
    }
}
