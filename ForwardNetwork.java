import com.sun.org.apache.regexp.internal.RE;
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
    RealMatrix W1 = p.generateRandomMatrix(inputLayerSize,hiddenLayerSize);
    RealMatrix W2 = p.generateRandomMatrix(hiddenLayerSize,outputLayerSize);
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

       // public HashMap<String,RealMatrix> costFunctionPrimeFF(RealMatrix yCaret, RealMatrix y, RealMatrix X) {
        public void costFunctionPrimeFF(RealMatrix yCaret, RealMatrix y, RealMatrix X) {
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
        delta2 = delta3.multiply(W2.transpose()).multiply(calculate.sigmoidPrime(z2));
        dJdW1 = X.transpose().multiply(delta2);

     //   HashMap h = new HashMap();
     //   h.put("dJdW1",dJdW1);
     //   h.put("dJdW2",dJdW2);

       // return h;
    }

    public double[] calculateGradient(RealMatrix X, RealMatrix y, RealMatrix yCaret) {
        //HashMap h = new HashMap();
        costFunctionPrimeFF(yCaret,y,X);
        double[] v1 = trash.matrixFlattener(dJdW1);
        double[] v2 = trash.matrixFlattener(dJdW2);

        return trash.vectorConcatenator(v1,v2);
    }

    public double[] calculateNumericalGradient(RealMatrix X, RealMatrix y) {
        System.out.println(W1);
        System.out.println(W2);
        double[] v1 = trash.matrixFlattener(W1);
        double[] v2 = trash.matrixFlattener(W2);
        double[] numericalGradient = trash.vectorConcatenator(v1,v2);
        for (int i = 0; i <= numericalGradient.length-1; i++) {
            System.out.println(numericalGradient[i]);
        }
        for (int i = 0; i <= numericalGradient.length - 1; i++) {
            numericalGradient[i] = 0;
        }
        double[] perturbation = numericalGradient.clone();
        for (int i = 0; i <= numericalGradient.length-1; i++) {
            System.out.println(perturbation[i]);
        }
        double e = 1e-4;

        for (int i = 0; i <= numericalGradient.length - 1; i++) {
            perturbation[i] = e;
            RealMatrix lossFunction2 = costFunctionFF(yCaret,y);
        //  !  System.out.println(lossFunction2);

            double[] uberVector = new double[perturbation.length];
            for (int j = 0; j <= perturbation.length - 1; j++) {
                uberVector[j] = numericalGradient[j] - perturbation[j];
            }
            trash.vectorCutter(uberVector,v1.length);
            v1 = trash.v1;
            v2 = trash.v2;
            W1 = trash.matrixReshaper(v1,W1.getRowDimension(),W1.getColumnDimension());
            W2 = trash.matrixReshaper(v2,W2.getRowDimension(),W2.getColumnDimension());
          //  W1 = trash.matrixReshaper(v1,W1.getColumnDimension(),W1.getRowDimension());
          //  W2 = trash.matrixReshaper(v2,W2.getColumnDimension(),W2.getRowDimension());

            RealMatrix lossFunction1 = costFunctionFF(yCaret,y);
           // numericalGradient[i] = (Double.parseDouble(lossFunction2.add(lossFunction1).scalarMultiply((-1)).scalarMultiply(1/(2*e)).getEntry(0,0).toString()));
            numericalGradient[i] = (lossFunction2.add(lossFunction1).scalarMultiply((-1)).scalarMultiply(1/(2*e)).getEntry(0,0));
         //d   System.out.println("pisiiiiii: " + lossFunction2.add(lossFunction1).scalarMultiply((-1)).scalarMultiply(1/(2*e)).getEntry(0,0));
           // lossFunction2.add(lossFunction1).scalarMultiply((-1)).scalarMultiply(1/(2*e));
           // System.out.println(lossFunction1);
            perturbation[i] = 0;

        }

        return numericalGradient;
    }











    }
