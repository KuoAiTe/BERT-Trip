/**
 * Created by Estrid on 13/12/2017.
 */
class TimeConverter {

    double getDateDiff(long firstTimeStamp, long secondTimeStamp)  {

        if(firstTimeStamp>secondTimeStamp){
            long tmp = firstTimeStamp;
            firstTimeStamp = secondTimeStamp;
            secondTimeStamp = tmp;
        }
        return (secondTimeStamp - firstTimeStamp)/60.0;

    }

}
