import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

/**
 * Created by Estrid on 14/09/2017.
 */
public class OrienteeringEnvironment {

    int numberOfPOIs;
    POI[] POIGraph;
    double[][] problemRepresentation;
    double timeBudget;
    int start;
    int end;
    private double[] avgVisitDurationForEachPOI;
    double[][] cooccurrTable;

    OrienteeringEnvironment(POI[] POIGraph, double[][] problemGraph, double[] avgVisitDurationForEachPOI) {
        this.problemRepresentation = problemGraph;
        this.numberOfPOIs = problemGraph.length;
        this.POIGraph = POIGraph;
        this.timeBudget = 0;
        this.start = 0;
        this.end = 0;
        this.avgVisitDurationForEachPOI = avgVisitDurationForEachPOI;
        for (int i = 0; i < numberOfPOIs; i++) {
            POIGraph[i].visitTime = avgVisitDurationForEachPOI[i];
        }
        this.cooccurrTable = new double[numberOfPOIs][numberOfPOIs];
    }

    void setConstraints(Visit[] testSequence, double timeBudget) {
        this.start = testSequence[0].POI;
        this.end = testSequence[testSequence.length-1].POI;
        this.timeBudget = timeBudget;

    }

    void setInterestScore(double[] userInterest){
        for (int i = 0; i<POIGraph.length; i++){
            POIGraph[i].setInterest(userInterest[i]);
        }

    }

    int getNumberOfPOIs() {
        return this.problemRepresentation.length;
    }

    void personalizeVisitDuration(double[] userInt) {
        for (int i = 0; i < POIGraph.length; i++) {
            if(userInt[i]>0.0) {
                POIGraph[i].visitTime = userInt[i] * this.avgVisitDurationForEachPOI[i];
            }
        }
    }

    void resetVisitDuration(){
        for (int i = 0; i < POIGraph.length; i++) {
            POIGraph[i].visitTime = this.avgVisitDurationForEachPOI[i];
        }
    }

    double getTransitCost(Integer POI1, Integer POI2){

        return this.problemRepresentation[POI1][POI2];
    }


    double getTimeCost(Sequence sequence){
        return getTimeCost(sequence.visits);
    }


    private double getTimeCost(Visit[] sequence){
        double totalCost = 0;
        for(int i = 0; i<sequence.length-1;i++){
            totalCost = totalCost + getTransitCost(sequence[i].POI, sequence[i+1].POI);
        }
        for(int i =1; i<sequence.length-1; i++){
            totalCost = totalCost + this.POIGraph[sequence[i].POI].visitTime;
        }
        return totalCost;
    }

    double getVisitDuration(Integer POI){
        return this.POIGraph[POI].visitTime;
    }


}
