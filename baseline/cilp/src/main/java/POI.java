import java.util.Random;

/**
 * Created by Estrid on 15/09/2017.
 */
public class POI {
    private int ID;
    int frequency;
    String category;
    private double latitude;
    private double longitude;
    private double popularity;
    double visitTime;
    double interest;
    private double[] latentVector;

    public POI(String ID, String frequency, String category, String latitude, String longitude) {
        this.ID = Integer.parseInt(ID)-1;
        this.frequency = Integer.parseInt(frequency);
        this.category = category;
        this.latitude = Double.parseDouble(latitude);
        this.longitude = Double.parseDouble(longitude);
        this.popularity = 0;
        this.interest = 0;
        this.visitTime = 1;
    }

    POI(String ID, String latitude, String longitude, String category){
        this.ID = Integer.parseInt(ID)-1;
        this.category = category;
        this.latitude = Double.parseDouble(latitude);
        this.longitude = Double.parseDouble(longitude);
        this.visitTime = 1;
    }
    public POI(String ID, String frequency){
        this.ID = Integer.parseInt(ID);
        this.frequency = Integer.parseInt(frequency);
        this.category = null;
        this.latitude = 0;
        this.longitude = 0;
        this.popularity = 0;
        this.interest = 0;
        this.visitTime = 1;

    }
    void setPop(double popularity){
        this.popularity = popularity;
    }
    void setInterest(double interest){
        this.interest = interest;
    }

    public void setVisitTime(double visitTime){
        this.visitTime=visitTime;
    }
    public double getDistance(POI b){
        final int R = 6371; // Radius of the earth


        double lat1 = this.latitude;
        double lon1 = this.longitude;
        double lat2 = b.latitude;
        double lon2 = b.longitude;

        double latDistance = Math.toRadians(lat2 - lat1);
        double lonDistance = Math.toRadians(lon2 - lon1);
        double x = Math.sin(latDistance / 2) * Math.sin(latDistance / 2)
                + Math.cos(Math.toRadians(lat1)) * Math.cos(Math.toRadians(lat2))
                * Math.sin(lonDistance / 2) * Math.sin(lonDistance / 2);
        double y = 2 * Math.atan2(Math.sqrt(x), Math.sqrt(1 - x));
        double distance = R * y ; // in kilometers

        return distance;
    }
    public void initializeLatentVector(int numOfLatentFactors){
        latentVector = new double[numOfLatentFactors];
        for (int i = 0; i < numOfLatentFactors; i++) {
            latentVector[i]=new Random().nextDouble();
        }
    }

}
