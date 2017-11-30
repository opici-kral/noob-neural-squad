import org.apache.commons.math3.linear.RealMatrix;

public class vypisovacMatrix2D {
    public void bleee(String name, RealMatrix matrix) {
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
}
