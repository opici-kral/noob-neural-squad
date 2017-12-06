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

        HashMap h = new HashMap();
        h.put("dJdW1",dJdW1);
        h.put("dJdW2",dJdW2);

       // return h;
    }









    }
