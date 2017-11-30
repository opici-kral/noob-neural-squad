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

        ForwardNetwork f = new ForwardNetwork();
     f.forward(X);

     RealMatrix W1 = f.W1;
     RealMatrix W2 = f.W2;
     RealMatrix z2 = f.z2;
     RealMatrix a2 = f.a2;
     RealMatrix z3 = f.z3;
     RealMatrix yCaret = f.yCaret;


     RealMatrix J = f.costFunctionFF(yCaret,y);
     HashMap h = f.costFunctionPrimeFF(yCaret,y,X);
        RealMatrix dJdW1 = f.dJdW1;
        RealMatrix dJdW2 = f.dJdW2;

        System.out.println("X: " + X);
        System.out.println("y: " + y);
        System.out.println("W1: " + W1);
        System.out.println("W2: " + W2);
        System.out.println("z1: " + z2);
        System.out.println("a2: " + a2);
        System.out.println("z3: " + z3);
        System.out.println("yCaret: " + yCaret);

        vypisovacMatrix2D plf = new vypisovacMatrix2D();
        plf.bleee("X",X);
        plf.bleee("y",y);
        plf.bleee("W1",W1);
        plf.bleee("W2",W2);
        plf.bleee("z1",z2);
        plf.bleee("a2",a2);
        plf.bleee("z3",z3);
        plf.bleee("yCaret",yCaret);
        plf.bleee("J",J);
        plf.bleee("dJdW1",dJdW1);
        plf.bleee("dJdW2",dJdW2);


    }
}
