import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public class PseudoRandomizer {

    public RealMatrix generateRandomMatrix(int inputLayerSize, int hiddenLayerSize) {
        RealMatrix matrix = MatrixUtils.createRealMatrix(inputLayerSize, hiddenLayerSize);
        for (int i = 0; i <= inputLayerSize - 1; i++) {
            for (int j = 0; j <= hiddenLayerSize - 1; j++) {
                matrix.setEntry(i, j, Math.pow(-1, Math.round(Math.random() * 10)) * Math.random() + Math.pow(-1, Math.round(Math.random() * 10)));
            }
        }
        return matrix;
    }
}
