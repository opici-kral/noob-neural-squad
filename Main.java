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
        System.out.println(n);

        RealMatrix W1 = f.W1;
        RealMatrix W2 = f.W2;
        double [] W1W2Flatt = t.vectorConcatenator(t.matrixFlattener(W1),t.matrixFlattener(W2));
        RealMatrix Xhat = o.naiveBFGS(W1W2Flatt, null, gradVector, X, y);
        t.bleeMatrix("Xhat", Xhat);







        /* RealMatrix J = f.costFunctionFF(yCaret,y);
     //HashMap h = f.costFunctionPrimeFF(yCaret,y,X);
     f.costFunctionPrimeFF(yCaret,y,X);
        RealMatrix dJdW1 = f.dJdW1;
        RealMatrix dJdW2 = f.dJdW2;
        TrashyThingie t = new TrashyThingie();
        double [] W1W2Flatt = t.vectorConcatenator(t.matrixFlattener(W1),t.matrixFlattener(W2));


        System.out.println("ddd: "  );


        t.bleeMatrix("X",X);
        t.bleeMatrix("y",y);
        t.bleeMatrix("W1",W1);
        t.bleeMatrix("W2",W2);
        t.bleeMatrix("z1",z2);
        t.bleeMatrix("a2",a2);
        t.bleeMatrix("z3",z3);
        t.bleeMatrix("yCaret",yCaret);
        t.bleeMatrix("J",J);
        t.bleeMatrix("dJdW1",dJdW1);
        t.bleeMatrix("dJdW2",dJdW2);
        System.out.println("W1W2");
        for (int i = 0; i <= W1W2Flatt.length - 1; i++) {
            System.out.print(W1W2Flatt[i] + " ");
        }
        System.out.println("");
        t.vectorCutter(W1W2Flatt,W1.getColumnDimension()*W1.getRowDimension());
        t.bleeVector("v1",t.v1);
        t.bleeVector("v2",t.v2);
        t.bleeVector("nabla",f.calculateGradient(X,y,yCaret));

        System.out.println("ted:");
        t.bleeVector("numgr: ",f.calculateNumericalGradient(X,y));
        f.callbackFunction(W1W2Flatt,y);*/
    }
}
