import org.apache.commons.math3.linear.RealMatrix;

public class ActivationFunction {
    public RealMatrix sigmoid(RealMatrix z) {
        int columnDimension = z.getColumn(0).length - 1;
        int rowDimension = z.getRow(0).length - 1;
        RealMatrix z1 = z.copy();
        for (int i = 0; i <= columnDimension; i++) {
            for (int j = 0; j <= rowDimension; j++) {
                z1.setEntry(i, j, (1/(1 + Math.exp((-1)*z.getEntry(i, j)))));
            }
        }
        return z1;
    }

    public RealMatrix sigmoidPrime(RealMatrix z) {
        int columnDimension = z.getColumn(0).length - 1;
        int rowDimension = z.getRow(0).length - 1;
        RealMatrix z1 = z.copy();
        for (int i = 0; i <= columnDimension; i++) {
            for (int j = 0; j <= rowDimension; j++) {
                z1.setEntry(i, j, (Math.exp((-1)*z.getEntry(i, j))/(1 + Math.pow(Math.exp((-1)*z.getEntry(i, j)),2))));
            }
        }
        return z1;
    }
}
