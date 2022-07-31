/**
 * Created by Estrid on 13/10/2017.
 */
public class User {
    public String id;
    public int numOfLatentFactors;
    public double[] latentVector;
    public int diminishParameter;

    public User(String id, int numOfLatentFactors, double[] latentVector, int diminishParameter) {
        this.id = id;
        this.numOfLatentFactors = numOfLatentFactors;
        this.latentVector = latentVector;
        this.diminishParameter = diminishParameter;
    }
}
