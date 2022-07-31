import java.io.*;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * Created by Estrid on 6/10/2017.
 */
class Preprocessor {
    private Path dir;
    private String visitFile;
    private String sequenceFile;
    private String poiFile;
    private String costPopFile;


    Preprocessor(Path dir, String city) {
        this.dir = dir;
        this.visitFile = Paths.get(this.dir.toString(), "userVisits-"+ city +".csv").toString();
        this.sequenceFile = Paths.get(this.dir.toString(), "sequence-"+city+".csv").toString();
        this.poiFile = Paths.get(this.dir.toString(), "POI-" + city + ".csv").toString();
        this.costPopFile = Paths.get(this.dir.toString(), "costProfCat-" + city + "POI-all.csv").toString();
    }

    double[][] getTimeCostTable(int numberOfPOIs)throws IOException{
        double[][] timeCostTable = new double[numberOfPOIs][numberOfPOIs];
        try (BufferedReader reader = new BufferedReader(new FileReader(this.costPopFile))) {
            reader.readLine();
            String line;
            while ((line = reader.readLine()) != null) {
                String[] tokens = line.split(";");
                int from = Integer.parseInt(tokens[0].replace("\"", ""))-1;
                int to = Integer.parseInt(tokens[1].replace("\"", ""))-1;
                double distanceCost = Double.parseDouble(tokens[2].replace("\"",""));
                double timeCost = distanceCost*0.015;
                timeCostTable[from][to] = timeCost;
            }
        }
        return timeCostTable;
    }

    POI[] getPOIGraphFromFile() throws IOException {
        List<POI> POIs = new LinkedList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(this.poiFile))) {
            reader.readLine();
            String line;
            while ((line = reader.readLine()) != null) {
                String[] tokens = line.split(";");
                POI poi = new POI (tokens[0], tokens[2], tokens[3], tokens[4]);// have made the POI ID begin from 0
                POIs.add(poi);
            }
        }
        return POIs.toArray(new POI[0]);

    }

    private Map<Integer, LinkedList<Double>> getVisitDuration(Map<String, LinkedList<Sequence>> allUserSequences) {
        Map<Integer, LinkedList<Double>> visitDurations = new HashMap<>();
        for(Map.Entry<String, LinkedList<Sequence>> entry: allUserSequences.entrySet()){
            LinkedList<Sequence> sequences = entry.getValue();
            for(Sequence sequence: sequences){
                for (int i = 0; i < sequence.length(); i++) {
                    int POI = sequence.visits[i].POI;
//                    double visitDuration = sequence.visits[i].departureTime-sequence.visits[i].arrivalTime;
                    double visitDuration = new TimeConverter().getDateDiff(sequence.visits[i].departureTime,sequence.visits[i].arrivalTime);
                    if(visitDuration != 0.0){
                        if(visitDurations.containsKey(POI)){
                            visitDurations.get(POI).add(visitDuration);
                        }else{
                            LinkedList<Double> list = new LinkedList<>();
                            list.add(visitDuration);
                            visitDurations.put(POI, list);
                        }
                    }
                }

            }
        }
        for(Map.Entry<Integer, LinkedList<Double>> entry: visitDurations.entrySet()){
            entry.getValue().sort(new Comparator<Double>() {
                @Override
                public int compare(Double o1, Double o2) {
                    return (o1.compareTo(o2));
                }
            });
        }
        return visitDurations;
    }

    private static double[] getAvgVisitTimeUsingMean(Map<Integer, LinkedList<Double>> mappedVisits, int numPOIs){
        double[] avgVisitTime = new double[numPOIs];
        for (int i = 0; i < numPOIs; i++) {
            if(mappedVisits.containsKey(i)){
                LinkedList<Double> visitList = mappedVisits.get(i);
                double totalTime = 0;
                for(Double visitDuration:visitList){
                    totalTime = totalTime + visitDuration;
                }
                double avgTime = totalTime/(visitList.size());
                    avgVisitTime[i] = avgTime;
            }else{
                avgVisitTime[i] = 1000.0;
            }
        }

        return avgVisitTime;
    }


    double[] updateVisitTime(Map<String, LinkedList<Sequence>> allUserSequences, POI[] POIGraph) {
        int numPOIs = POIGraph.length;
        Map<Integer, LinkedList<Double>> visitDurationForEachPOI= getVisitDuration(allUserSequences);
        return getAvgVisitTimeUsingMean(visitDurationForEachPOI, numPOIs);
    }


    void updatePopularity(POI[] POIGraph, Map<String, LinkedList<Sequence>> userSequences){
        Integer[] visitCount = getOverallPOICount(userSequences, POIGraph.length);
        for(int i = 0; i<visitCount.length; i++){
            POIGraph[i].frequency = visitCount[i];
        }

        int maxFrequency = 0;

        for (POI poi : POIGraph) {
            maxFrequency = Math.max(maxFrequency, poi.frequency);
        }
        for (POI poi : POIGraph) {
            poi.setPop((double) poi.frequency / maxFrequency + 0.000001);
        }
    }

    private Integer[] getOverallPOICount(Map<String, LinkedList<Sequence>> userSequences, int numberOfPOIS){
        Integer[] count = new Integer[numberOfPOIS];
        Arrays.fill(count, 0);

        for (Map.Entry<String, LinkedList<Sequence>> entry : userSequences.entrySet()) {
            for (Sequence value : entry.getValue()) {
                Visit[] sequence = value.visits;
                for (Visit visit : sequence) {
                    count[visit.POI]++;
                }
            }
        }
        return count;
    }

    Map<String,LinkedList<Sequence>> getTestUserSequences(Map<String, LinkedList<Sequence>> allUserSequences, int minSeqLen, int minSeqPerUser) {
        Map<String, LinkedList<Sequence>> filteredSequences = new HashMap<>();
        for(Map.Entry<String, LinkedList<Sequence>> entry: allUserSequences.entrySet()){
            LinkedList<Sequence> sequenceList = entry.getValue();
            if(sequenceList.size()>= minSeqPerUser){
                LinkedList<Sequence> testSequences = new LinkedList<>();
                for(Sequence sequence: sequenceList){
                    if(sequence.length()>=minSeqLen){
                        testSequences.add(sequence);
                    }
                }
                if(!testSequences.isEmpty()){
                    filteredSequences.put(entry.getKey(),testSequences);
                }
            }
        }
        return filteredSequences;
    }

    Map<String,LinkedList<Sequence>> getUserSequences() throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(new File(this.sequenceFile)));
        Map<String, LinkedList<Sequence>> userSequences = new HashMap<>();
        String nextLine;
        while((nextLine = reader.readLine())!=null){
            String[] strings = nextLine.split(" ");
            String user = strings[0];
            int number = Integer.parseInt(strings[1]);
            int length = (strings.length-2)/3;
            Visit[] visits = new Visit[length];
            for (int i = 0; i < length; i++) {
                visits[i] = new Visit(Integer.parseInt(strings[2+i*3]), Long.parseLong(strings[3+i*3]), Long.parseLong(strings[4 + 3*i]));
            }
            Sequence sequence = new Sequence(visits,number);
            if(userSequences.containsKey(user)){
                userSequences.get(user).add(sequence);
            }else{
               LinkedList<Sequence> newEntry = new LinkedList<>();
               newEntry.add(sequence);
               userSequences.put(user, newEntry);
            }
        }
        reader.close();
        return userSequences;
    }

    private void writeUserSequences() throws IOException {
        Map<String, LinkedList<Sequence>> userSequences = new HashMap<>();
        BufferedReader reader = new BufferedReader(new FileReader(new File(this.visitFile)));
        String tmp;

        reader.readLine();


        String[] firstLine = reader.readLine().split(";");
        int currentPOI = Integer.parseInt(firstLine[3])-1;
        long time = Long.parseLong(firstLine[2]);
        String user = firstLine[1].replace("\"","");
        firstLine[6]= firstLine[6].replace("\"", "");
        Integer currentSeq = Integer.valueOf(firstLine[6]);
        Sequence state = new Sequence(new Visit(currentPOI,time, time), currentSeq);
        while((tmp = reader.readLine())!=null){
            String[] tokens = tmp.split(";");
            Integer POI = Integer.parseInt(tokens[3])-1;
            time = Long.parseLong(tokens[2]);
            Integer seq = Integer.valueOf(tokens[6].replaceAll("\"", ""));
            if(currentSeq.equals(seq) && !POI.equals(currentPOI)){
                currentPOI = POI;
                if(!state.contains(currentPOI)){
                    state.add(new Visit(currentPOI, time, time));
                    int index = state.getIndexOf(currentPOI);
                    state.visits[index].departureTime = time;
                }
            }else if(currentSeq.equals(seq) && POI.equals(currentPOI)){
                state.visits[state.length()-1].departureTime = Math.max(state.visits[state.length()-1].departureTime, time);
            }else if (!seq.equals(currentSeq)){
                if (userSequences.containsKey(user)) {
                    List<Sequence> theUserSequences = userSequences.get(user);
                    theUserSequences.add(state);
                } else {
                    LinkedList<Sequence> theUSerSequences = new LinkedList<>();
                    theUSerSequences.add(state);
                    userSequences.put(user, theUSerSequences);
                }
                state = new Sequence(new Visit(POI, time, time), seq);
                currentPOI = POI;
                currentSeq = seq;
                user = tokens[1].replace("\"","");
            }
        }
        int seqCount = 0;
        for(Map.Entry<String, LinkedList<Sequence>> entry:userSequences.entrySet()){
            seqCount = seqCount + entry.getValue().size();
        }
        System.out.println("Total sequences: "+ seqCount);
        int visitCount = 0;
        for(Map.Entry<String, LinkedList<Sequence>> entry:userSequences.entrySet()){
            for(Sequence sequence: entry.getValue()){
                visitCount = visitCount + sequence.length();
            }

        }
        System.out.println("Total visits: "+ visitCount);

        Map<String, LinkedList<Sequence>> resultSequences = replaceUserID(userSequences);

        BufferedWriter writer = new BufferedWriter(new FileWriter(new File(this.sequenceFile)));
        int count = 0;
        for(Map.Entry<String, LinkedList<Sequence>> entry: resultSequences.entrySet()){

            for(Sequence sequence: entry.getValue()){
                writer.write(entry.getKey() + " " +sequence.number + " ");
                for (int i = 0; i < sequence.visits.length; i++) {
                    writer.write(sequence.visits[i].POI + " " +sequence.visits[i].arrivalTime + " " + sequence.visits[i].departureTime + " ");
                }
                count ++;
                writer.newLine();
            }

        }
        System.out.println(count);
        writer.close();
    }

    private Map<String, LinkedList<Sequence>> replaceUserID (Map<String, LinkedList<Sequence>> userSequences){
        int i =0;
        Map<String, LinkedList<Sequence>> newUserSequences = new HashMap<>();
        for(Map.Entry<String, LinkedList<Sequence>> entry: userSequences.entrySet()){
            LinkedList<Sequence> oldEntry = entry.getValue();
            newUserSequences.put(String.valueOf(i), oldEntry);
            i++;
        }
        return newUserSequences;
    }

    private void writeUserAllPOIsTrainingFile(Map<String, LinkedList<Sequence>> userSequences, Map<String, LinkedList<Sequence>> testSequences) throws IOException {
        String path = Paths.get(this.dir.toString(), "userAllPOIsFile").toString();
        for(Map.Entry<String, LinkedList<Sequence>> testUser: testSequences.entrySet()){
            for(Sequence testSequence: testUser.getValue()){
                BufferedWriter writer = new BufferedWriter(new FileWriter(new File(Paths.get(path, testSequence.number+".txt").toString())));
                for(Map.Entry<String, LinkedList<Sequence>> entry: userSequences.entrySet()){
                    String user = entry.getKey();
                    writer.write(user + " ");
                    LinkedList<Integer> mergedList = new LinkedList<>();
                    for(Sequence sequence: entry.getValue()){
                        if(sequence.number!=testSequence.number) {
                            for (int i = 0; i < sequence.length(); i++) {
                                if(!mergedList.contains(sequence.visits[i].POI)){
                                    mergedList.add(sequence.visits[i].POI);
                                }

                            }
                        }
                    }
                    for(Integer poi: mergedList){
                        writer.write(poi + " ");
                    }
                    writer.newLine();
                }
                writer.close();
            }
        }
    }

    private void writeUserAllCheckinsTrainingFile(Map<String, LinkedList<Sequence>> userSequences, Map<String, LinkedList<Sequence>> testSequences) throws IOException {
        String path = Paths.get(this.dir.toString(), "userAllCheckinsFile").toString();
        for(Map.Entry<String, LinkedList<Sequence>> testUser: testSequences.entrySet()){
            for(Sequence testSequence: testUser.getValue()){
                BufferedWriter writer = new BufferedWriter(new FileWriter(new File(Paths.get(path, testSequence.number+".txt").toString())));
                for(Map.Entry<String, LinkedList<Sequence>> entry: userSequences.entrySet()){
                    String user = entry.getKey();
                    writer.write(user + " ");
                    for(Sequence sequence: entry.getValue()){
                        if(sequence.number!=testSequence.number) {
                            for (int i = 0; i < sequence.length(); i++) {
                                writer.write(sequence.visits[i].POI + " ");
                            }
                        }
                    }

                    writer.newLine();
                }
                writer.close();
            }
        }
    }

    private void writeUserSequencesTrainingFile(Map<String, LinkedList<Sequence>> userSequences, Map<String, LinkedList<Sequence>> testSequences) throws IOException {
        String path = Paths.get(this.dir.toString(), "userSequencesFile").toString();
        for(Map.Entry<String, LinkedList<Sequence>> testUser: testSequences.entrySet()){
            for(Sequence testSequence: testUser.getValue()){
                BufferedWriter writer = new BufferedWriter(new FileWriter(new File(Paths.get(path, testSequence.number+".txt").toString())));
                for(Map.Entry<String, LinkedList<Sequence>> entry: userSequences.entrySet()){
                    String user = entry.getKey();
                    for(Sequence sequence: entry.getValue()){
                        // make sure training data doesn't contain test sequence
                        if(sequence.number!=testSequence.number) {
                            writer.write(user + " ");
                            for (int i = 0; i < sequence.length(); i++) {
                                writer.write(sequence.visits[i].POI + " ");
                            }
                            writer.newLine();
                        }
                    }
                }
                writer.close();
            }
        }

    }
    public void process(String city, String dataDir, int minSeqPerUser, int minSeqLen) throws IOException {
        Path currentPath = Paths.get(System.getProperty("user.dir"));
        Path filePath = Paths.get(currentPath.toString(), dataDir, city);
        Preprocessor preprocessor = new Preprocessor(filePath, city);
        preprocessor.writeUserSequences();
        Map<String, LinkedList<Sequence>> userSequences = preprocessor.getUserSequences();
        Map<String, LinkedList<Sequence>> testSequences = preprocessor.getTestUserSequences(userSequences,minSeqLen, minSeqPerUser);
        preprocessor.writeUserSequencesTrainingFile(userSequences, testSequences);
        preprocessor.writeUserAllCheckinsTrainingFile(userSequences, testSequences);
        preprocessor.writeUserAllPOIsTrainingFile(userSequences,testSequences);
    }


}
