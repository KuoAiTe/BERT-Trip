import lpsolve.LpSolve;
import lpsolve.LpSolveException;
import java.util.LinkedList;

/**
 * Created by Estrid on 21/02/2018.
 */
public class ContextualILPSolver{
    private int numVariables;
    private int start;
    private int end;
    private double budget;
    private int numPOIs;

    private long timeout;
    private OrienteeringEnvironment environment;
    ContextualILPSolver(int numPOIs, OrienteeringEnvironment environment, long timeout) {
        this.numVariables = numPOIs*numPOIs+numPOIs;
        this.numPOIs = numPOIs;
        this.start = environment.start;
        this.end = environment.end;
        this.budget = environment.timeBudget + environment.POIGraph[this.end].visitTime;
        this.environment = environment;
        this.timeout = timeout;
        this.numVariables = numPOIs*numPOIs + numPOIs*numPOIs + numPOIs;
        // compute the profitTable after all parameters are updated.
    }

    private double[][] initiateMatrix(int numPOI) {
        return new double[this.numPOIs*2 + 1][this.numPOIs];
    }

    private double[] getNewProfitTable(OrienteeringEnvironment environment) {
        double[][] newTable = initiateMatrix(this.numPOIs);
        for (int i = 0; i < this.numPOIs; i++) {
            for (int j = 0; j < this.numPOIs; j++) {
                newTable[i][j] = environment.POIGraph[i].interest;
            }
        }
        for (int i = 0; i < this.numPOIs; i++) {
            for (int j = 0; j < this.numPOIs; j++) {
                newTable[i+this.numPOIs][j] = environment.cooccurrTable[i][j]*0;
            }
        }
        double[] profit = toVector(newTable);
        double[] profitVector = new double[profit.length+1];
        for (int i = 0; i < profit.length; i++) {
            profitVector[i+1] = profit[i];
        }
        return profitVector;
    }

    private double[][] toMatrix(double[] vector) {
        double[][] matrix = new double[vector.length/this.numPOIs][this.numPOIs];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                matrix[i][j] = vector[i*matrix[0].length+j];
            }
        }
        return matrix;
    }

    private double[] toVector(double[][] matrix){

        double[] vector = new double[matrix.length*matrix[0].length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                vector[matrix[0].length*i+j] = matrix[i][j];
            }
        }
        return vector;
    }

    private double[] getNewTimeCostTable(OrienteeringEnvironment environment) {
        double[][] originalTimeCostTable = environment.problemRepresentation;
        double[][] newTimeCostTable = initiateMatrix(this.numPOIs);
        for (int i = 0; i < this.numPOIs; i++) {
            for (int j = 0; j < this.numPOIs; j++) {
                newTimeCostTable[i][j] = originalTimeCostTable[i][j] + environment.POIGraph[j].visitTime;
            }
        }
        return toVector(newTimeCostTable);
    }

    private void resetMatrix(double[][] matrix){
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                matrix[i][j] = 0.0;
            }
        }
    }

    private int[] initiateColNo(int numVariables){
        int[] colno = new int[numVariables];
        for (int i = 0; i < numVariables; i++) {
            colno[i] = i+1;
        }
        return colno;
    }

    Sequence solve() throws LpSolveException {
        double[] profitVector = getNewProfitTable(this.environment);
        LpSolve solver = LpSolve.makeLp(0,this.numVariables);

        int[] colno = initiateColNo(this.numVariables);


        for (int i = 0; i < this.numPOIs*this.numPOIs*2; i++) {
            solver.setInt(i + 1, true);
        }

        solver.setAddRowmode(true);

        //startPOI
        double[][] variableMatrix = initiateMatrix(this.numPOIs);
        for (int i = 0; i < this.numPOIs; i++) {
            variableMatrix[this.start][i] = 1;
        }
        variableMatrix[this.start][this.start] = 0;
        double[] constraint1 = toVector(variableMatrix);
        solver.addConstraintex(this.numVariables, constraint1, colno, LpSolve.EQ, 1);


        resetMatrix(variableMatrix);
        for (int i = 0; i < numPOIs; i++) {
            variableMatrix[i][start] = 1;
        }
        variableMatrix[start][start] = 0;
        double[] constraint2 = toVector(variableMatrix);
        solver.addConstraintex(this.numVariables, constraint2, colno, LpSolve.EQ, 0);

        //end point

        resetMatrix(variableMatrix);
        for (int i = 0; i < this.numPOIs; i++) {
            variableMatrix[this.end][i] = 1;
        }
        double[] constraint3 = toVector(variableMatrix);
        solver.addConstraintex(this.numVariables, constraint3, colno, LpSolve.EQ, 0);


        resetMatrix(variableMatrix);
        for (int i = 0; i < this.numPOIs; i++) {
            variableMatrix[i][this.end] = 1;
        }
        double[] constraint4 = toVector(variableMatrix);
        solver.addConstraintex(this.numVariables, constraint4, colno, LpSolve.EQ, 1);


        //other points
        for (int i = 0; i < this.numPOIs; i++) {
            resetMatrix(variableMatrix);
            if (i != this.start && i != this.end) {
                for (int j = 0; j < this.numPOIs; j++) {
                    variableMatrix[i][j] = 1;
                }
                double[] constraint5 = toVector(variableMatrix);
                solver.addConstraintex(this.numVariables, constraint5, colno, LpSolve.LE, 1);
                for (int j = 0; j < this.numPOIs; j++) {
                    variableMatrix[j][i] = -1;
                }
                variableMatrix[i][i]=0;
                double[] constraint6 = toVector(variableMatrix);
                solver.addConstraintex(this.numVariables, constraint6, colno, LpSolve.EQ, 0);
            }
        }

        resetMatrix(variableMatrix);
        for (int i = 0; i < this.numPOIs; i++) {
            variableMatrix[i][i] = 1; // Two consecutive points cannot be same
        }
        double[] constraint7 = toVector(variableMatrix);
        solver.addConstraintex(this.numVariables, constraint7, colno, LpSolve.EQ, 0);


        //time budget
        double[] timeCostVector = getNewTimeCostTable(this.environment);
        solver.addConstraintex(this.numVariables, timeCostVector, colno, LpSolve.LE, this.budget);

        //Eliminate sub-tours
        for (int i = 0; i < numPOIs; i++) {
            if(i!=start) {
                resetMatrix(variableMatrix);
                variableMatrix[this.numPOIs*2][i] = 1.0;
                double[] constraint8 = toVector(variableMatrix);
                solver.addConstraintex(this.numVariables, constraint8, colno, LpSolve.LE, this.numPOIs);
                solver.addConstraintex(this.numVariables, constraint8, colno, LpSolve.GE, 2);
            }
        }

        for (int i = 0; i < this.numPOIs; i++) {
            if(i!=this.start) {
                for (int j = 0; j < this.numPOIs; j++) {
                    if(j!=this.start){
                        resetMatrix(variableMatrix);
                        variableMatrix[this.numPOIs*2][i] = 1.0;
                        variableMatrix[this.numPOIs*2][j] = -1.0;
                        variableMatrix[i][j] = this.numPOIs-1;
                        double[] constraint9 = toVector(variableMatrix);
                        solver.addConstraintex(this.numVariables, constraint9, colno, LpSolve.LE, this.numPOIs-2);
                    }
                }

            }
        }

        for (int i = 0; i < this.numPOIs; i++) {
            if (i != this.end) {
                for (int j = 0; j < this.numPOIs; j++) {
                    if(j!=this.end) {
                        resetMatrix(variableMatrix);
                        for (int k = 0; k < this.numPOIs; k++) {
                            variableMatrix[i][k] = -1;
                        }
                        variableMatrix[this.numPOIs + i][j] = 1;
                        double[] constraint10 = toVector(variableMatrix);
                        solver.addConstraintex(this.numVariables, constraint10, colno, LpSolve.LE, 0);
                    }
                }
            }
        }

        for (int i = 0; i < this.numPOIs; i++) {
            if(i!=this.end) {
                for (int j = 0; j < this.numPOIs; j++) {
                    if(j!=this.end) {
                        resetMatrix(variableMatrix);
                        for (int k = 0; k < this.numPOIs; k++) {
                            variableMatrix[j][k] = -1;
                        }
                        variableMatrix[this.numPOIs + i][j] = 1;
                        double[] constraint11 = toVector(variableMatrix);
                        solver.addConstraintex(this.numVariables, constraint11, colno, LpSolve.LE, 0);
                    }
                }
            }
        }

        for (int i = 0; i < this.numPOIs; i++) {
            if(i!=this.end) {
                for (int j = 0; j < this.numPOIs; j++) {
                    if(j!=this.end) {
                        resetMatrix(variableMatrix);
                        variableMatrix[this.numPOIs + i][j] = 1;
                        for (int k = 0; k < this.numPOIs; k++) {
                            variableMatrix[i][k] = -1;
                        }
                        for (int k = 0; k < this.numPOIs; k++) {
                            variableMatrix[j][k] = -1;
                        }
                        double[] constraint11 = toVector(variableMatrix);
                        solver.addConstraintex(this.numVariables, constraint11, colno, LpSolve.GE, -1);
                    }
                }
            }
        }

        for (int i = 0; i < this.numPOIs; i++) {
            resetMatrix(variableMatrix);
            variableMatrix[i+this.numPOIs][this.end] = 1;
            for (int j = 0; j < this.numPOIs; j++) {
                variableMatrix[i][j] = -1;
            }
            double[] constraint12 = toVector(variableMatrix);
            solver.addConstraintex(this.numVariables,constraint12,colno, LpSolve.EQ, 0);
        }

        for (int i = 0; i < this.numPOIs; i++) {
            resetMatrix(variableMatrix);
            variableMatrix[i+this.numPOIs][this.end] = 1;
            variableMatrix[this.end+this.numPOIs][i] = -1;
            double[] constraint13 = toVector(variableMatrix);
            solver.addConstraintex(this.numVariables,constraint13,colno, LpSolve.EQ, 0);
        }



        solver.setAddRowmode(false);

        solver.setObjFn(profitVector);
        solver.setMaxim();


        solver.setVerbose(LpSolve.IMPORTANT);
        solver.setTimeout(this.timeout);
        int isSolved = solver.solve();

        if(isSolved == 7){
            return null; // If a TIMEOUT occurs, return null
        }else {

            double[] var = solver.getPtrVariables();
            double objectiveOri = solver.getObjective();
            System.out.println("Value of objective function: " +
                    objectiveOri);

            double[][] resultMatrix = toMatrix(var);
            LinkedList<Integer> result = new LinkedList<>();
            int selectedPOI = start;
            result.add(selectedPOI);
            while (selectedPOI != end) {
                for (int i = 0; i < numPOIs; i++) {
                    if (resultMatrix[selectedPOI][i] != 0) {
                        selectedPOI = i;
                        result.add(selectedPOI);
                        break;
                    }
                }
            }
            Sequence solution = new Sequence(result.toArray(new Integer[0]));
            // delete the problem and free memory
            solver.deleteLp();
            return solution;
        }
    }
}
