import com.sun.org.apache.regexp.internal.RE;
import org.apache.commons.math3.linear.DiagonalMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.*;

public class OptimizationThingie {
    public double epsilon = 1e-5;

    public RealMatrix naiveBFGS(double[] weightsVector, RealMatrix costFunction, double[] gradientVector, RealMatrix X, RealMatrix y) {
        TrashyThingie t = new TrashyThingie();
        ForwardNetwork f = new ForwardNetwork();

        double[] x0 = weightsVector;

        double[] g0 = gradientVector;
       // t.bleeVector("gradientVector",gradientVector);
        RealMatrix H = t.createIdentityMatrix(weightsVector.length);
        RealMatrix H0 = new LUDecomposition(H).getSolver().getInverse();
        RealMatrix G0 = t.matrixReshaper(g0,g0.length,1);
        RealMatrix X0 = t.matrixReshaper(x0,x0.length,1);
        RealMatrix X1 = X0.copy();
        int i = 1;

        while (G0.getNorm() >= epsilon) {

            H0 = new LUDecomposition(H0).getSolver().getInverse();
            RealMatrix H0G0 = H0.multiply(G0);
            X1 = X0.add(H0G0.scalarMultiply(-1));
            double[] x1 = t.matrixFlattener(X1);
            int cutLength = f.W1.getColumnDimension() * f.W1.getRowDimension();
            t.vectorCutter(x1, cutLength);
            f.W1 = t.matrixReshaper(t.v1, f.W1.getRowDimension(), f.W1.getColumnDimension());
            f.W2 = t.matrixReshaper(t.v2, f.W2.getRowDimension(), f.W2.getColumnDimension());
            RealMatrix yCaret = f.forward(X);
            double[] g1 = f.calculateGradient(X, y, yCaret);

            double[] ydif = t.vectorAppend(g1, t.vectorScalarMultiply(g0, -1));

            double[] sdif = t.vectorAppend(x1, t.vectorScalarMultiply(x0, -1));
            RealMatrix Ydif = t.matrixReshaper(ydif, 1, ydif.length);
            RealMatrix Sdif = t.matrixReshaper(sdif, sdif.length, 1);
            RealMatrix YtS = Ydif.multiply(Sdif);
            double yts = YtS.getEntry(0, 0);
            RealMatrix H1 = H0.copy();
            if (yts > 0) {
                RealMatrix A = (Ydif.transpose().multiply(Ydif)).scalarMultiply((1 / yts));
                RealMatrix B = (H0.multiply(Sdif.multiply(Sdif.transpose())).multiply(H0)).scalarMultiply(1 / (Sdif.transpose().multiply((H0).multiply(Sdif)).getEntry(0, 0)));
                H1 = (H0.add(A)).add(B.scalarMultiply(-1));
              //  t.bleeMatrix("H0",H0);
            } else {
                H1 = H0;
            }
            G0 = t.matrixReshaper(g1, g1.length, 1);
            X0 = X1;
            H0 = H1;
            g0 = t.matrixFlattener(G0);
            x0 = t.matrixFlattener(X0);
            i++;
        }






      //  t.bleeVector("g1",g1);
      //  t.bleeVector("g0",g0);
      //  t.bleeVector("ydif",ydif);
      //  t.bleeVector("sdif",sdif);
       // t.bleeMatrix("Ydif",Ydif);
       // t.bleeMatrix("Sdif",Sdif);
      //  t.bleeMatrix("Sdif",Sdif);
     //   t.bleeMatrix("YtS",YtS);
     //   t.bleeMatrix("A",A);
     //   t.bleeMatrix("B",B);

        //t.bleeMatrix("W1",f.W1);
        //t.bleeMatrix("W2",f.W2);


        //     t.vectorCutter();

     //   RealMatrix yCaret = f.forward(X);
    //    f.costFunctionPrimeFF(X,y,yCaret);

       // f.W1 = f.W1.add(f.dJdW1.scalarMultiply(-3));
       // f.W2 = f.W2.add(f.dJdW2.scalarMultiply(-3));
       // f.forward(X);
      //  yCaret = f.yCaret;
      //  RealMatrix yCaret = f.forward(X1);
      //  double[] g1 = f.calculateGradient(X,y,yCaret);



     //   x1 = x0




      //  t.bleeVector("weightsVector",weightsVector);
      //  t.bleeVector("gradientVector", gradientVector);
      //  t.bleeMatrix("H0",H0);
      //  t.bleeMatrix("H0",H0);
      //  t.bleeMatrix("G0",G0);
     //   t.bleeMatrix("H0G0",H0G0);
     //   t.bleeMatrix("X1",X1);
    //    t.bleeVector("g1",g1);


        System.out.println("zdeees:");
        System.out.println(f.forward(X));

        return X1;
    }
}
