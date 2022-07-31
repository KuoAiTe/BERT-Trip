import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;

/**
 * Created by Estrid on 18/09/2017.
 */
public class Sequence {
    Visit[] visits;
    private double timeBudget;
    int number;

    Sequence(Visit visit, int number){
        this.visits = new Visit[1];
        this.visits[0] = visit;
        this.number = number;
    }

    Sequence(Visit[] visits, int number){
        this.visits = visits;
        this.number = number;
    }

    Sequence(Integer[] sequence){
        this.visits = new Visit[sequence.length];
        for (int i = 0; i < sequence.length; i++) {
            this.visits[i] = new Visit(sequence[i]);
        }
        this.timeBudget = 0;
    }

    Sequence(){
        this.visits = new Visit[0];
    }

    int length(){
        return this.visits.length;
    }

    private float getRecall(Sequence originalSequence){
        int count = 0;
        for(int i = 0; i< this.length();i++){
            if(originalSequence.contains(this.visits[i].POI)){
                count ++;
            }
        }
        return (float)count/(originalSequence.length());
    }

    private float getPrecision(Sequence originalSequence){
        int count = 0;
        for(int i = 0; i< this.length();i++){
            if(originalSequence.contains(this.visits[i].POI)){
                count ++;
            }
        }
        return (float)count/this.length();
    }

    private float getPrecision2(Sequence sequence){
        if(this.length() == 2){
            return 0;
        }else{
            int count = 0;
            for (int i = 1; i < this.length() - 1; i++) {
                if(sequence.contains(this.visits[i].POI)){
                    count ++;
                }
            }
            return (float)count/(this.length() - 2);
        }
    }

    private float getRecall2(Sequence sequence){
        if(this.length() == 2){
            return 0;
        }else{
            int count = 0;
            for (int i = 1; i < this.length() - 1; i++) {
                if(sequence.contains(this.visits[i].POI)){
                    count ++;
                }
            }
            return (float)count/(sequence.length() - 2);
        }
    }

    private float getFScore(float precision, float recall){
        if(precision == 0 && recall == 0){
            return 0;
        }else{
            return 2*precision*recall/(precision + recall);
        }
    }

    boolean contains(Integer poi){
        for(int i = 0; i< length(); i++){
            if(visits[i].POI ==poi){
                return true;
            }
        }
        return false;
    }


    @Override
    public boolean equals(Object obj) {
        Sequence seq = (Sequence)obj;
        if(seq.visits.length!=this.length()){
            return false;
        }else {
            for (int i = 0; i < length(); i++) {
                if (this.visits[i].POI != ((Sequence) obj).visits[i].POI) {
                    return false;
                }
            }
        }
        return true;
    }

    @Override
    public String toString() {
        String st = String.valueOf(visits[0].POI);
        for (int i = 1; i < visits.length; i++){
            st = st.concat(","+visits[i].POI);
        }
        return st;
    }

    String toString(OrienteeringEnvironment environment) {
        String st = String.valueOf(visits[0].POI);
        st = st.concat("("+environment.POIGraph[visits[0].POI].category+")");
        for (int i = 1; i < visits.length; i++){
            st = st.concat("-->"+visits[i].POI+"("+environment.POIGraph[visits[i].POI].category+") ["+ String.valueOf(new TimeConverter().getDateDiff(visits[i].arrivalTime,visits[i].departureTime))+"]");
        }
        return st;
    }

    int getIndexOf(int currentPOI) {
        for (int i = 0; i < length(); i++) {
            if(visits[i].POI == currentPOI){
                return i;
            }
        }
        return -1;
    }

    void add(Visit visit) {

        Visit[] originalVisits = this.visits;
        this.visits = new Visit[length()+1];
        for (int i = 0; i < originalVisits.length; i++) {
            this.visits[i] = originalVisits[i];
        }
        this.visits[length()-1] = visit;
    }

    Map<String,Float> getResultStats(Sequence sequence, String[] stats) {
        Map<String, Float> resultStats = new HashMap<>();
        float precision = getPrecision(sequence);
        float recall = getRecall(sequence);
        float precision_2 = getPrecision2(sequence);
        float recall_2 = getRecall2(sequence);
        for (int i = 0; i < stats.length; i++) {
            String stat = stats[i];
            switch (stat){
                case "precision":
                  resultStats.put("precision", precision);
                  System.out.println("precision = " + precision);
                break;
                case "recall":
                  resultStats.put("recall", recall);
                  System.out.println("recall = " + recall);
                break;
                case "F1":
                  resultStats.put("F1", getFScore(precision, recall));
                  System.out.println("F1 = " + getFScore(precision, recall));
                break;
                case"precision_2":
                  resultStats.put("precision_2", precision_2);
                  System.out.println("precision_2 = " + precision_2);
                break;
                case "recall_2":
                  resultStats.put("recall_2", recall_2);
                  System.out.println("recall_2 = " + precision_2);
                  break;
                case "F1_2":
                  resultStats.put("F1_2", getFScore(precision_2,recall_2));
                  System.out.println("F1_2 = " + getFScore(precision_2,recall_2));
                  break;
            }
        }
        return resultStats;
    }
}
