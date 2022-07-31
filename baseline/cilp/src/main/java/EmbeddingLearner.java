import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Created by Estrid on 6/02/2018.
 */
class EmbeddingLearner {
    private Path path;
    private int maxIterations;
    private double[][] userMatrix;
    private double[] userBias;
    private double[][] itemMatrix;
    private double[] itemBias;
    private int numOfDim;
    private LinkedList<Integer[]> dataset; // for each input, the first element is the userID, the rest are the POIs in the sequence
    private int numOfItems;
    private double learnRate;
    private double regBias;
    private double regItem;
    private Map<Integer, LinkedList<Integer>> userVisitedPOIs;
    private int testSequenceNumber;
    private double contextWeight;


    EmbeddingLearner(Path path, int testSequenceNumber, int maxIterations, int numOfUsers, int numOfItems, int numOfDim, double learnRate, double reg, double contextWeight) {
        this.path = path;
        this.maxIterations = maxIterations;
        this.numOfItems = numOfItems;
        this.numOfDim = numOfDim;
        this.dataset = new LinkedList<>();
        this.userVisitedPOIs = new HashMap<>();
        this.learnRate = learnRate;
        this.userMatrix = randomMatrix(numOfUsers);
        this.itemMatrix = randomMatrix(this.numOfItems);
        this.itemBias = randomVector(this.numOfItems);
        this.regBias = reg;
        this.regItem = reg;
        this.testSequenceNumber = testSequenceNumber;
        this.contextWeight = contextWeight;
    }

    private double[] randomVector(int numOfItems){
        double[] vector = new double[numOfItems];
        for (int i = 0; i < vector.length; i++) {
            vector[i] = ThreadLocalRandom.current().nextDouble();
        }
        return vector;
    }

    private double[][] randomMatrix(int numOfItems){
        double[][] matrix = new double[numOfItems][this.numOfDim];
        for (int i = 0; i < numOfItems; i++) {
            for (int j = 0; j < this.numOfDim; j++) {
                matrix[i][j] = ThreadLocalRandom.current().nextDouble();
            }
        }
        return matrix;
    }


    private void loadData(String path) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(new File(path)));
        String text;
        while((text = reader.readLine())!=null){
            String[] strings = text.split(" ");
            Integer[] sequence = new Integer[strings.length];
            for (int i = 0; i < strings.length; i++) {
                sequence[i] = Integer.valueOf(strings[i]);
            }
            this.dataset.add(sequence);
        }
    }

    private void loadUserVisitedPOIs(String path) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(new File(path)));
        String text;
        while((text = reader.readLine())!=null){
            String[] strings = text.split(" ");
            Integer user = Integer.valueOf(strings[0]);
            LinkedList<Integer> list = new LinkedList<>();
            for (int i = 1; i < strings.length; i++) {
                list.add(Integer.valueOf(strings[i]));
            }
            this.userVisitedPOIs.put(user, list);
        }
    }

    private double logisticGradient (double diff){
        return 1/(1 + Math.exp(-diff));
    }

    private double multiple(double[] vec1, double[] vec2){
        double product = 0.0;
        for (int i = 0; i < vec1.length; i++) {
            product += vec1[i]*vec2[i];
        }
        return product;
    }

    private double[] update(double[] vec1, double gradient, double[] vec2){
        double[] result = new double[vec1.length];
        for (int i = 0; i < vec1.length; i++) {
            result[i] = vec1[i] + this.learnRate * ( gradient * vec2[i] - 2*this.regItem * vec1[i]);
        }
        return result;
    }



    private double[] minus(double[] vec1, double[] vec2){
        double[] result = new double[vec1.length];
        for (int i = 0; i < vec1.length; i++) {
            result[i] = vec1[i] - vec2[i];
        }
        return result;
    }





    private Integer[] computeContextPOIs(Integer[] sequence, int index){
        Integer[] contextPOIs = new Integer[sequence.length - 2];
        int j = 0;
        for (int i = 1; i < sequence.length; i++) {
            if(i!=index){
                contextPOIs[j] = sequence[i];
                j++;
            }
        }
        return contextPOIs;
    }

    private double[] aggregate(Integer[] contextPOIs, double[] vec2){
        double[] result = new double[this.numOfDim];

        for (int i = 0; i < result.length; i++) {
            result[i] += vec2[i];
            for (int j = 0; j < contextPOIs.length; j++) {
                result[i] += this.itemMatrix[contextPOIs[j]][i]*this.contextWeight;
            }
        }
        return result;
    }

    private boolean contains(Integer[] list, Integer key){
        for (int i = 0; i < list.length; i++) {
            if(list[i].equals(key)){
                return true;
            }
        }
        return false;
    }

    private void updateWithoutContext (Integer user, Integer[] sequence){
        for (int i = 1; i < sequence.length; i++) {
            for (int j = 0; j < this.numOfItems; j++) {
                if (j!=sequence[i]) {
                    double gradient = 1 - logisticGradient(multiple(this.userMatrix[user], this.itemMatrix[sequence[i]]) + this.itemBias[sequence[i]] - multiple(this.userMatrix[user], this.itemMatrix[j]) - this.itemBias[j]);

                    double[] userFactors = copy(this.userMatrix[user]);
                    double[] itemFactorsi = copy(this.itemMatrix[sequence[i]]);
                    double[] itemFactorsj = copy(this.itemMatrix[j]);
                    this.userMatrix[user] = copy(update(userFactors, gradient, minus(itemFactorsi, itemFactorsj)));
                    this.itemMatrix[sequence[i]] = copy(update(itemFactorsi, gradient, userFactors));
                    this.itemMatrix[j] = copy(update(itemFactorsj, -gradient, userFactors));
                    this.itemBias[sequence[i]] = this.itemBias[sequence[i]] + this.learnRate * (gradient - 2*this.regBias * this.itemBias[sequence[i]]);
                    this.itemBias[j] = this.itemBias[j] + this.learnRate * (-gradient - 2*this.regBias * this.itemBias[j]);
                }
            }
        }
    }

    private void trainWithContextual(){
        int iter = 0;
        Collections.shuffle(this.dataset);
        while(iter<this.maxIterations){
            iter++;
            for(Integer[] sequence: this.dataset){
                Integer user = sequence[0];
                if(sequence.length < 3){
                    updateWithoutContext(user, sequence);
                } else{
                    for (int i = 1; i < sequence.length; i++) {
                        Integer[] contextPOIs = computeContextPOIs(sequence, i);
                        double[] userContextVector = aggregate(contextPOIs, this.userMatrix[user]);
                        for (int j = 0; j < this.numOfItems; j++) {
                            if (sequence[i] != j && !contains(contextPOIs, j)) {
                                double gradient = 1 - logisticGradient(multiple(userContextVector, this.itemMatrix[sequence[i]]) + this.itemBias[sequence[i]]
                                        - multiple(userContextVector, this.itemMatrix[j]) - this.itemBias[j] );
                                double[] userFactors = copy(this.userMatrix[user]);
                                double[] itemFactorsi = copy(this.itemMatrix[sequence[i]]);
                                double[] itemFactorsj = copy(this.itemMatrix[j]);
                                this.userMatrix[user] = copy(update(userFactors, gradient, minus(itemFactorsi, itemFactorsj)));
                                this.itemMatrix[sequence[i]] = copy(update(itemFactorsi, gradient, userContextVector));
                                this.itemMatrix[j] = copy(update(itemFactorsj, -gradient, userContextVector));
                                this.itemBias[sequence[i]] = this.itemBias[sequence[i]] + this.learnRate * (gradient - 2*this.regBias * this.itemBias[sequence[i]]);
                                this.itemBias[j] = this.itemBias[j] + this.learnRate * (-gradient - 2*this.regBias * this.itemBias[j]);
                                for (int k = 0; k < contextPOIs.length; k++) {
                                    double[] vectorOfOneContextPOI = copy(this.itemMatrix[contextPOIs[k]]);
                                    this.itemMatrix[contextPOIs[k]] = copy(update(vectorOfOneContextPOI, gradient*this.contextWeight, minus(itemFactorsi, itemFactorsj)));
                                }

                            }
                        }
                    }
                }

            }
        }
    }


    private double[] copy(double[] vec){
        double[] result = new double[vec.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = vec[i];
        }
        return result;
    }

    double[] predictWithContextual(String testUser, OrienteeringEnvironment environment) throws IOException {
        // for each test sequence, load its corresponding training data
        loadData(Paths.get(this.path.toString(), "userSequencesFile", this.testSequenceNumber + ".txt").toString());
        loadUserVisitedPOIs(Paths.get(this.path.toString(), "userAllPOIsFile", this.testSequenceNumber + ".txt").toString());
        trainWithContextual();
        double[] userInt = new double[this.numOfItems];
        double z = 0.0;
        Integer[] contextPOIs = {environment.start, environment.end};
        double[] userContextVector = aggregate(contextPOIs, this.userMatrix[Integer.parseInt(testUser)]);
        for (int i = 0; i < this.itemMatrix.length; i++) {
            userInt[i] = multiple(userContextVector, this.itemMatrix[i])+this.itemBias[i];
            z += Math.exp(userInt[i]);
        }
        for (int i = 0; i < userInt.length; i++) {
            userInt[i] = Math.exp(userInt[i])/z;
        }


        double z2 = 0;
        for (int i = 0; i < this.numOfItems; i++) {
            if (i != environment.start && i != environment.end){
                for (int j = 0; j < this.itemMatrix.length; j++) {
                    if (i != j && j != environment.start && j != environment.end) {
                        environment.cooccurrTable[i][j] = multiple(this.itemMatrix[i], this.itemMatrix[j]);
                    z2 += Math.exp(environment.cooccurrTable[i][j]);
                    }
                }
            }

        }


        for (int i = 0; i < this.numOfItems; i++) {
            if(i!=environment.start && i!=environment.end) {
                for (int j = 0; j < this.numOfItems; j++) {
                    if (i != j && j != environment.start && j != environment.end) {
                    environment.cooccurrTable[i][j] = Math.exp(environment.cooccurrTable[i][j]) / z2;;
                    }
                }
            }
        }


        for (int i = 0; i < userInt.length; i++) {
            System.out.print("["+i+"] "+ userInt[i]+" ");
        }

        System.out.println();
        return userInt;

    }

}
