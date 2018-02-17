import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.Arrays;

public class PseudoNormalizer {

    public RealMatrix Normalize(RealMatrix matrix) {
        int numberOfColumnsX = matrix.getRow(0).length - 1;
        for (int i = 0; i <= numberOfColumnsX; i++) {
            double[] ithColumn = matrix.getColumn(i);
            double max = Arrays.stream(ithColumn).max().getAsDouble();
            System.out.println(max);
            RealMatrix C = MatrixUtils.createColumnRealMatrix(ithColumn);
            C = C.scalarMultiply(1 / max);
            ithColumn = C.getColumn(0);
            matrix.setColumn(i, ithColumn);
        }
        return matrix;
    }

    public RealMatrix NormalizeClassic(RealMatrix matrix, double[] matrixArray) {
        double positiveMax = Arrays.stream(matrixArray).max().getAsDouble();
        double negativeMin = Arrays.stream(matrixArray).filter(a -> a < 0).min().orElse(0);
        double max = Math.max(positiveMax,Math.abs(negativeMin));

      //  System.out.println(Math.abs(negativeMin) + " " + positiveMax );
        for (int i = 0; i <= matrix.getRowDimension() - 1; i++) {
            for (int j = 0; j <= matrix.getColumnDimension() - 1; j++) {
                matrix.setEntry(i,j,matrix.getEntry(i,j)/max);
            }
        }
        return matrix;
    }
}
