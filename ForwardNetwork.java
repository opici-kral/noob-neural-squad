import com.sun.org.apache.regexp.internal.RE;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;


import java.util.HashMap;
import java.util.List;

public class ForwardNetwork {
    int inputLayerSize = 2;
    int hiddenLayerSize = 3;
    int outputLayerSize = 1;
    RealMatrix z2;
    RealMatrix a2;
    RealMatrix z3;
    RealMatrix yCaret;
    RealMatrix J;
    RealMatrix delta3;
    RealMatrix dJdW2;
    RealMatrix delta2;
    RealMatrix dJdW1;

    PseudoRandomizer p = new PseudoRandomizer();
   // RealMatrix W1 = p.generateRandomMatrix(inputLayerSize,hiddenLayerSize);
   // RealMatrix W2 = p.generateRandomMatrix(hiddenLayerSize,outputLayerSize);
    RealMatrix W1 = new Array2DRowRealMatrix(new double[][] { {1.0904269362,-1.8807253547,1.1897603268},{-0.2841332172,-1.7588658304,1.7520965746} }, false); //p.generateRandomMatrix(inputLayerSize,hiddenLayerSize);
    RealMatrix W2 = new Array2DRowRealMatrix(new double[][] { {0.1360694199},{-0.0785684103},{0.1459850139} }, false);//p.generateRandomMatrix(hiddenLayerSize,outputLayerSize);
    TrashyThingie trash = new TrashyThingie();




    public RealMatrix forward(RealMatrix X) {
        ActivationFunction calculate = new ActivationFunction();
        z2 = X.multiply(W1);
        a2 = calculate.sigmoid(z2);
        z3 = a2.multiply(W2);
        yCaret = calculate.sigmoid(z3);
        return yCaret;
    }

        public RealMatrix costFunctionFF(RealMatrix yCaret, RealMatrix y) {
            RealMatrix whyDelta = y.add(yCaret.scalarMultiply(-1));
            J = (whyDelta.transpose().multiply(whyDelta)).scalarMultiply(0.5);
            return J;
        }

        public void costFunctionPrimeFF(RealMatrix X, RealMatrix y, RealMatrix yCaret) {
        ActivationFunction calculate = new ActivationFunction();
        RealMatrix whyDelta = y.add(yCaret.scalarMultiply(-1));
        delta3 = MatrixUtils.createRealMatrix(whyDelta.getColumn(0).length,whyDelta.getRow(0).length);
        /* lambdas? alert, might be wrong, for extended output recalculate on paper! */
        for (int i = 0; i <= whyDelta.getColumn(0).length - 1; i++) {
            for (int j = 0; j <= whyDelta.getRow(0).length - 1; j++) {
                double deltaij = whyDelta.getEntry(i,j)*calculate.sigmoidPrime(z3).getEntry(i,j)*(-1);
                delta3.setEntry(i,j,deltaij);
            }
        }
        dJdW2 = a2.transpose().multiply(delta3);
        RealMatrix delta3DotW2T = delta3.multiply(W2.transpose());
        RealMatrix sigmoidPrimez2 = calculate.sigmoidPrime(z2);
        delta2 = z2.copy();
            for (int i = 0; i <= z2.getRowDimension() - 1; i++) {
                for (int j = 0; j <= z2.getColumnDimension() - 1; j++) {
                    double elementij = delta3DotW2T.getEntry(i,j)*sigmoidPrimez2.getEntry(i,j);
                    delta2.setEntry(i, j, elementij); //= delta3.multiply(W2.transpose()).multiply(calculate.sigmoidPrime(z2));
                }
            }
        dJdW1 = X.transpose().multiply(delta2);
    }

    public double[] calculateGradient(RealMatrix X, RealMatrix y, RealMatrix yCaret) {
        costFunctionPrimeFF(X,y,yCaret);
        double[] v1 = trash.matrixFlattener(dJdW1);
        double[] v2 = trash.matrixFlattener(dJdW2);
        return trash.vectorConcatenator(v1,v2);
    }

    public double[] calculateNumericalGradient(RealMatrix X, RealMatrix y) {
        double[] v1 = trash.matrixFlattener(W1);
        double[] v2 = trash.matrixFlattener(W2);
        double[] numericalGradient = trash.vectorConcatenator(v1,v2);
        double[] weights = numericalGradient.clone();

        for (int i = 0; i <= numericalGradient.length - 1; i++) {
            numericalGradient[i] = 0;
        }
        double[] perturbation = numericalGradient.clone();

        double e = 1e-4;

        for (int i = 0; i <= numericalGradient.length - 1; i++) {
            perturbation[i] = e;

            double[] uberVector = new double[perturbation.length];
            for (int j = 0; j <= perturbation.length - 1; j++) {
                uberVector[j] = weights[j] + perturbation[j];
            }

            trash.vectorCutter(uberVector,v1.length);
            v1 = trash.v1;
            v2 = trash.v2;
            W1 = trash.matrixReshaper(v1,W1.getRowDimension(),W1.getColumnDimension());
            W2 = trash.matrixReshaper(v2,W2.getRowDimension(),W2.getColumnDimension());
            forward(X);
            RealMatrix lossFunction2 = costFunctionFF(yCaret,y);

            for (int j = 0; j <= perturbation.length - 1; j++) {
                uberVector[j] = weights[j] - perturbation[j];
            }

            trash.vectorCutter(uberVector,v1.length);
            v1 = trash.v1;
            v2 = trash.v2;
            W1 = trash.matrixReshaper(v1,W1.getRowDimension(),W1.getColumnDimension());
            W2 = trash.matrixReshaper(v2,W2.getRowDimension(),W2.getColumnDimension());
            forward(X);
            RealMatrix lossFunction1 = costFunctionFF(yCaret,y);
            numericalGradient[i] = (lossFunction2.getEntry(0,0) - lossFunction1.getEntry(0,0))/(2*e);  //((lossFunction2.add((lossFunction1).scalarMultiply((-1)))).scalarMultiply(1/(2*e)).getEntry(0,0));
           //System.out.println("i:" + i );
           //System.out.println("=============");
           //System.out.println("loss2 " + lossFunction2);
           //System.out.println("loss1 " + lossFunction1);
           //System.out.println("2*e " + 2*e);
            // for (int g = 0; g <= numericalGradient.length - 1; g++) {
           //     System.out.print(numericalGradient[g] + " ");
           // }
           // System.out.println("");

            perturbation[i] = 0;

        }

        return numericalGradient;
    }


    public void callbackFunction(double[] uberVector, RealMatrix y) {
        double[] v1 = trash.matrixFlattener(W1);
        double[] v2 = trash.matrixFlattener(W2);
        trash.vectorCutter(uberVector,v1.length);
        v1 = trash.v1;
        v2 = trash.v2;
        W1 = trash.matrixReshaper(v1,W1.getRowDimension(),W1.getColumnDimension());
        W2 = trash.matrixReshaper(v2,W2.getRowDimension(),W2.getColumnDimension());
        J.add(costFunctionFF(yCaret,y));
        System.out.println(J);
    }
}
