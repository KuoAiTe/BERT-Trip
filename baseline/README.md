# C-ILP [e09fc6d](https://github.com/Estrid0112/context_aware_trip_recommendation)
### Run
    mvn install
    java -jar target/context_aware_trip_recommendation-1.0-SNAPSHOT-jar-with-dependencies.jar config/configuration.properties

###  Install
You'll need to install lpsolve5.5 and Java SDK 11
# DeepTrip [d12c3d8](https://github.com/gcooq/DeepTrip/commit/d12c3d8aeb744d59a7d19e3c934d36a018ca8035)

###  Install
    git clone
    conda create --name deeptrip python=3.6.9
    conda activate deeptrip
    pip3 install tensorflow-gpu==1.14.0
    pip3 install scikit-learn

> Don't install tensorflow-gpu using conda; otherwise, it won't work.

### Modification
Note that the original DeepTrip evaluates the results in each epoch and takes the best result among the epochs while doing leave-one-out cross validation(LOOCV). Only one trip query is being evaluated per training.

### Run
    conda activate deeptrip
    python3 deeptrip/run.py

# SelfTrip [dbdeb1b](https://github.com/gcooq/SelfTrip/commit/dbdeb1b4c3446baddfb1b948b99b0b14781ff7d8)

###  Install
    conda create -n selftrip python=3.9 tensorflow-gpu
    pip install python-dateutil
    pip install scikit-learn

###  Run
    conda activate selftrip
    python poi_embeddings.py
    python train.py


# Run Markov, Rank+Markov, MarkovPath, Rank+MarkovPath [5f748ed](https://github.com/computationalmedia/tour-cikm16)
The original source code is available at https://github.com/computationalmedia/tour-cikm16.

### Run
    ipython3 rank_markov.py
