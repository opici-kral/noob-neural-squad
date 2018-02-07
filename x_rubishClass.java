public class x_rubishClass {
    /** TrashyThingie t = new TrashyThingie();
     ForwardNetwork f = new ForwardNetwork();
     f.forward(X);
     RealMatrix W1 = f.W1;
     RealMatrix W2 = f.W2;
     RealMatrix yCaret = f.yCaret;
     t.bleeMatrix("W1",W1);
     t.bleeMatrix("W2",W2);
     RealMatrix J1 = f.costFunctionFF(yCaret,y);
     f.costFunctionPrimeFF(yCaret,X,y);
     RealMatrix dJdW1 = f.dJdW1;
     RealMatrix dJdW2 = f.dJdW2;
     t.bleeMatrix("dJdW1",dJdW1);
     t.bleeMatrix("dJdW2",dJdW2);
     double scalar = 3;
     f.W1 = f.W1.add(f.dJdW1.scalarMultiply(-3));
     f.W2 = f.W2.add(f.dJdW2.scalarMultiply(-3));
     f.forward(X);
     yCaret = f.yCaret;
     RealMatrix J2 = f.costFunctionFF(yCaret,y);
     t.bleeMatrix("J1",J1);
     t.bleeMatrix("J2",J2);**/

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
