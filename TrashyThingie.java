import org.apache.commons.math3.linear.RealMatrix;

public class TrashyThingie {

    public double[] matrixFlattener(RealMatrix m) {
        int matrixLength = m.getColumnDimension() * m.getRowDimension();
        int k = 0;
        System.out.println(m.getColumnDimension());
        System.out.println(m.getRowDimension());
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

    public double[] vectorConcatenator(double[] v1, double[] v2) {
        int vectorLength = v1.length + v2.length;
        double[] v = new double[vectorLength];
        int k = 0;
        for (int i = 0; i <= v1.length - 1; i++) {
            v[i] = v1[i];
            k = i + 1;
            //System.out.println("k["+k+"]");
        }
        for (int j = 0; j <= v2.length - 1; j++) {
            v[k + j] = v2[j];
           // System.out.println("k["+(k+j)+"]");
        }
        return v;
    }
}

