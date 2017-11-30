import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.Arrays;

public class PseudoNormalizer {

    public RealMatrix Normalize(RealMatrix matrix) {
        int numberOfColumnsX = matrix.getRow(0).length - 1;
        for (int i = 0; i <= numberOfColumnsX; i++) {
            double[] ithColumn = matrix.getColumn(i);
            double max = Arrays.stream(ithColumn).max().getAsDouble();
            RealMatrix C = MatrixUtils.createColumnRealMatrix(ithColumn);
            C = C.scalarMultiply(1 / max);
            ithColumn = C.getColumn(0);
            matrix.setColumn(i, ithColumn);
        }
        return matrix;
    }
}
