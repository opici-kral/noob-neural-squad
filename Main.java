import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.*;

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

        System.out.println(t.zeroAjzer(t.binarizer(11),8));

        PseudoRandomizer pseudoRandom = new PseudoRandomizer();
        int inputLayerSize = 2;
        int hiddenLayerSize = 16;
        int outputLayerSize = 1;
        int binaryDimension = 8;

        RealMatrix preSynapse_0 = pseudoRandom.generateRandomMatrix(inputLayerSize,hiddenLayerSize);
        RealMatrix preSynapse_1 = pseudoRandom.generateRandomMatrix(hiddenLayerSize,outputLayerSize);
        RealMatrix preSynapse_h = pseudoRandom.generateRandomMatrix(hiddenLayerSize,hiddenLayerSize);

        RealMatrix synapse_0_zero = matrix.NormalizeClassic(preSynapse_0,t.matrixFlattener(preSynapse_0));
        RealMatrix synapse_1_zero = matrix.NormalizeClassic(preSynapse_1,t.matrixFlattener(preSynapse_1));
        RealMatrix synapse_h_zero = matrix.NormalizeClassic(preSynapse_h,t.matrixFlattener(preSynapse_h));

        RealMatrix synapse_0 = synapse_0_zero.copy();
        RealMatrix synapse_1 = synapse_1_zero.copy();
        RealMatrix synapse_h = synapse_h_zero.copy();

        t.zeroTurner(synapse_0_zero);
        t.zeroTurner(synapse_1_zero);
        t.zeroTurner(synapse_h_zero);

        t.bleeMatrix("synapse_0",synapse_0);
        t.bleeMatrix("synapse_1",synapse_1);
        t.bleeMatrix("synapse_h",synapse_h);

        t.bleeMatrix("synapse_0_zero",synapse_0_zero);
        t.bleeMatrix("synapse_1_zero",synapse_1_zero);
        t.bleeMatrix("synapse_h_zero",synapse_h_zero);

        //starts for j in 10000
        int a = (int) (Math.random()*(256/2));
        int b = (int) (Math.random()*(256/2));
        int c = a + b;
        List<Integer> aL = t.zeroAjzer(t.binarizer(a),binaryDimension);
        List<Integer> bL = t.zeroAjzer(t.binarizer(b),binaryDimension);
        List<Integer> cL = t.zeroAjzer(t.binarizer(c),binaryDimension);
        List<Integer> dL = new ArrayList<Integer>(Collections.nCopies(8, 0));
        double error = 0;
        //RealMatrix layer2deltas List or matrix???
        double[][] layer1valuesVector ={{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};
        RealMatrix layer1values = MatrixUtils.createRealMatrix(layer1valuesVector);

        // for position in bindim
        int position = 0;
        double[][] vecX ={{aL.get(binaryDimension-position-1),bL.get(binaryDimension-position-1)}};
        double[][] vecY ={{cL.get(binaryDimension-position-1)}};
        RealMatrix X_ = MatrixUtils.createRealMatrix(vecX);
        RealMatrix y_ = MatrixUtils.createRealMatrix(vecY).transpose();


        System.out.println(" " + aL);
        System.out.println("+" + bL);
        System.out.println("--------------------------");
        System.out.println(" " + cL);
        System.out.println(a + " + " + b + " = " + c);
        System.out.println(" ");
        System.out.println(dL);
        System.out.println(layer1values);
        t.bleeMatrix("X",X_);
        t.bleeMatrix("y",y_);









    }
}
