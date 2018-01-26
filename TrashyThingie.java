import org.apache.commons.math3.linear.DiagonalMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.HashMap;

public class TrashyThingie {
    public double[] v1;
    public double[] v2;

    public double[] matrixFlattener(RealMatrix m) {
        int matrixLength = m.getColumnDimension() * m.getRowDimension();
        int k = 0;
        double[] vector = new double[matrixLength];
        if (m.getRowDimension() == 1) {
            for (int i = 0; i <= m.getColumnDimension() - 1; i++) {
                vector[i] = m.getEntry(0,i);
            }
        }
        if (m.getColumnDimension() == 1) {
            for (int i = 0; i <= m.getRowDimension() - 1; i++) {
                vector[i] = m.getEntry(i,0);
            }
        } else {

            for (int i = 0; i <= m.getRowDimension() - 1; i++) {
                for (int j = 0; j <= m.getColumnDimension() - 1; j++) {
                    vector[j + k] = m.getEntry(i, j);
                }
                k = m.getRowDimension() * (i + 1) + 1;
            }

        }
        return vector;
    }

    public double[] vectorConcatenator(double[] vector1, double[] vector2) {
        int vectorLength = vector1.length + vector2.length;
        double[] v = new double[vectorLength];
        int k = 0;
        for (int i = 0; i <= vector1.length - 1; i++) {
            v[i] = vector1[i];
            k = i + 1;
        }
        for (int j = 0; j <= vector2.length - 1; j++) {
            v[k + j] = vector2[j];
        }
        return v;
    }

    public void vectorCutter (double[] vector, int cutLenth) {
        v1 = new double[cutLenth];
        v2 = new double[vector.length - cutLenth];
        HashMap h = new HashMap();
        for (int i = 0; i <= cutLenth - 1; i++) {
            v1[i] = vector[i];
        }
        for (int i = 0; i <= vector.length - cutLenth - 1; i++) {
            v2[i] = vector[cutLenth + i];
        }
    }

    public RealMatrix matrixReshaper(double[] vector,int k, int l) {
        RealMatrix m = MatrixUtils.createRealMatrix(k,l);
        for (int i = 0; i <= k - 1; i++) {
            for (int j = 0; j <= l - 1; j++) {
                m.setEntry(i,j,vector[j+l*(i)]);
            }
    }
    return m;
}

    public void bleeMatrix(String name, RealMatrix matrix) {
        System.out.println(name + ":");
        if (matrix != null) {
            for (int i = 0; i <= matrix.getColumn(0).length - 1; i++) {
                System.out.println(matrix.getRowVector(i));
            }
            System.out.println("");
        }
        else {
            System.out.println("null");
        }
    }

    public void bleeVector(String name, double[] vector) {
        System.out.println(name + ":");
        if (vector != null) {
            System.out.print("[");
            for (int i = 0; i <= vector.length - 1; i++) {
                System.out.print(vector[i] + " ");
            }
            System.out.print("]");
            System.out.println("");
        }
        else {
            System.out.println("null");
        }
    }

    public double gradToNumgradTest(double[] grad, double[] numgrad, RealMatrix X, RealMatrix y) {
        ForwardNetwork f = new ForwardNetwork();
        RealMatrix yCaret = f.forward(X);
        double[] gradVector = f.calculateGradient(X,y,yCaret);
        double[] numgradVector = f.calculateNumericalGradient(X,y);
        double norma1 = 0;
        double norma2 = 0;
        for (int i = 0; i <= gradVector.length - 1; i++) {
            double a = Math.pow((gradVector[i] - numgradVector[i]),2);
            double b = Math.pow((gradVector[i] + numgradVector[i]),2);
            norma1 = norma1 + a;
            norma2 = norma2 + b;
        }
        double norm1 = Math.sqrt(norma1);
        double norm2 = Math.sqrt(norma2);
        return norm1/norm2;
    }

    public RealMatrix createIdentityMatrix(int dimension) {
        RealMatrix E = new DiagonalMatrix(dimension);
        for (int i = 0; i <= dimension - 1; i++) {
            E.setEntry(i, i, 1);
        }
        return E;
    }

    public double[] vectorAppend(double[] vectorA, double[] vectorB) {
        double[] vector = vectorA.clone();
        for (int i = 0; i <= vectorA.length - 1; i++) {
            vector[i] = vectorA[i] + vectorB[i];
        }
        return vector;
    }

    public double[] vectorScalarMultiply(double[] vector, double scalar) {
        for (int i = 0; i <= vector.length - 1; i++) {
            vector[i] = vector[i]*scalar;
        }
        return vector;
    }

}