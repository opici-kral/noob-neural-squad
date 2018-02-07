import com.sun.org.apache.regexp.internal.RE;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.Arrays;
import java.util.HashMap;

public class Main {
    public static void main(String[] args) {

        double[][] vectorX ={{3,5},{5,1},{10,2}};
        double[][] vectorY ={{75},{82},{93}};
        RealMatrix X = MatrixUtils.createRealMatrix(vectorX);
        RealMatrix y = MatrixUtils.createRealMatrix(vectorY);

        PseudoNormalizer matrix = new PseudoNormalizer();
        X = matrix.Normalize(X);
        y = y.scalarMultiply(0.01);


        TrashyThingie t = new TrashyThingie();
        ForwardNetwork f = new ForwardNetwork();
        OptimizationThingie o = new OptimizationThingie();

        RealMatrix yCaret = f.forward(X);
        double[] gradVector = f.calculateGradient(X,y,yCaret);
        double[] numgradVector = f.calculateNumericalGradient(X,y);
        double n = t.gradToNumgradTest(gradVector,numgradVector,X,y);
        System.out.println("grad2numgradNorm[" + n + "]");
        RealMatrix W1 = f.W1;
        RealMatrix W2 = f.W2;
        double [] W1W2Flatt = t.vectorConcatenator(t.matrixFlattener(W1),t.matrixFlattener(W2));
        RealMatrix Xhat = o.naiveBFGS(W1W2Flatt, null, gradVector, X, y);
        t.bleeMatrix("Xhat", Xhat);

        System.out.println(t.binarizer(4));





    }
}
