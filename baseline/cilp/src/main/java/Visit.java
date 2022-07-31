/**
 * Created by Estrid on 11/12/2017.
 */
public class Visit {
    public int POI;
    public long arrivalTime;
    public long departureTime;

    public Visit(int POI, long arrivalTime, long departureTime) {
        this.POI = POI;
        this.arrivalTime = arrivalTime;
        this.departureTime = departureTime;
    }

    public Visit(int POI) {
        this.POI = POI;
    }
}
