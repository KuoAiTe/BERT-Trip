import lpsolve.LpSolveException;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedList;
import java.util.Map;
import java.util.Properties;
import java.util.logging.LogManager;

/**
 * Created by Estrid on 13/09/2017.
 */
public class Main {
    public static void main(String[] args) throws IOException, LpSolveException{
        LogManager.getLogManager().reset();

        String configFile = args[0];

        // load the properties file:
        FileReader reader = new FileReader(new File(configFile));
        Properties props = new Properties();
        props.load(reader);

        /*
          Set parameters
         */
        int minSeqPerUser = Integer.parseInt(props.getProperty("minSeqPerUser"));
        int minSeqLength = Integer.parseInt(props.getProperty("minSeqLen"));;

        /*
         * Read data
         */
        String city = props.getProperty("city");
        Path currentPath = Paths.get(System.getProperty("user.dir"));
        Path filePath = Paths.get(currentPath.toString(), props.getProperty("dataDir"), city);
        String dir = filePath.toString();
        Preprocessor preprocessor = new Preprocessor(filePath, city);
        preprocessor.process(city, props.getProperty("dataDir"), minSeqPerUser, minSeqLength);

        Map<String, LinkedList<Sequence>> allUserSequences = preprocessor.getUserSequences();
        Map<String, LinkedList<Sequence>> testSequences = preprocessor.getTestUserSequences(allUserSequences, minSeqLength, minSeqPerUser);

        POI[] POIGraph = preprocessor.getPOIGraphFromFile();

        double[][] timeCostTable = preprocessor.getTimeCostTable(POIGraph.length);

        preprocessor.updatePopularity(POIGraph, allUserSequences);
        // time cost in minutes
        double[] avgVisitDurationForEachPOI = preprocessor.updateVisitTime(allUserSequences, POIGraph);
        OrienteeringEnvironment environment = new OrienteeringEnvironment(POIGraph, timeCostTable, avgVisitDurationForEachPOI);

        Recommender recommender = new Recommender(filePath, testSequences, allUserSequences, environment, props);
        recommender.recommend();
    }
}
