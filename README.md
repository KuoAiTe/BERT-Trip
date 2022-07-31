## Code
Our implementation is based on [Hugging Face](https://huggingface.co/).

## Baselines
All of the investigated baselines were obtained from authors' Github repositories. The baselines are listed in the following:
- **Markov, Rank+Markov, MarkovPath, Rank+MarkovPath** [https://github.com/computationalmedia/tour-cikm16](https://github.com/computationalmedia/tour-cikm16)
 - **DeepTrip** [https://github.com/gcooq/DeepTrip](https://github.com/gcooq/DeepTrip)
 - **SelfTrip** [https://github.com/gcooq/SelfTrip](https://github.com/gcooq/SelfTrip)
 - **C-ILP** [https://github.com/Estrid0112/context_aware_trip_recommendation](https://github.com/Estrid0112/context_aware_trip_recommendation)

## Why was there a discrepancy between the experimental results in our paper and the results in prior work?
The authors [2] (Section VII-A) pointed out the experimental evaluation in prior studies [1], [3] included two known points (**origin**, **destination**) in the computation of <img src="https://render.githubusercontent.com/render/math?math=F_{1}"> and <img src="https://render.githubusercontent.com/render/math?math=Pairs-F_{1}"> scores. These two points are the inputs and should not be considered in terms of the predictive performance. Work in [4], [5] also used this metric.

We deliberately evaluated the Flickr dataset using this metric with all incorrect predictions and obtained surprisingly high <img src="https://render.githubusercontent.com/render/math?math=F_{1}"> scores. The <img src="https://render.githubusercontent.com/render/math?math=F_{1}"> scores supposed to be **zero** were instead evaluated as **0.529 (Edinburgh), 0.597 (Glasgow), 0.595 (Osaka), 0.518 (Melbourne), 0.580 (Toronto)**. In our experiments, we excluded the trip origin and the trip destination in the computation of <img src="https://render.githubusercontent.com/render/math?math=F_{1}"> , <img src="https://render.githubusercontent.com/render/math?math=Pairs-F_{1}">, and BLEU scores. Thus, this is why we had a great discrepancy between the experimental results in our paper and the results in prior work. A code example can be found at [https://github.com/KuoAiTe/BERT-Trip/blob/main/toy_example/metric.ipynb](https://github.com/KuoAiTe/BERT-Trip/blob/main/toy_example/metric.ipynb) or [https://colab.research.google.com/drive/17b8YGloP_ZurKJNbknAQ5fjEbVhCZhQp?usp=sharing](https://colab.research.google.com/drive/17b8YGloP_ZurKJNbknAQ5fjEbVhCZhQp?usp=sharing)
All their code is available on their Github repositories. You are welcome to test this metric by yourself. We also present a toy example using this metric in the following.

## Metric
You can check the code in the following:

https://github.com/computationalmedia/tour-cikm16/blob/master/parse_results.ipynb (function calc_F1)

https://github.com/gcooq/DeepTrip/blob/master/Trip/metric.py (lines 3-26)

https://github.com/gcooq/SelfTrip/blob/main/metric.py (lines 6-29)

```
import numpy as np
def calc_F1(traj_act, traj_rec, noloop=False):
    '''Compute recall, precision and F1 for recommended trajectories'''
    assert(isinstance(noloop, bool))
    assert(len(traj_act) > 0)
    assert(len(traj_rec) > 0)

if noloop == True:
    intersize = len(set(traj_act) & set(traj_rec))
else:
    match_tags = np.zeros(len(traj_act), dtype=np.bool)
    for poi in traj_rec:
        for j in range(len(traj_act)):
            if match_tags[j] == False and poi == traj_act[j]:
                match_tags[j] = True
                break
    intersize = np.nonzero(match_tags)[0].shape[0]

recall = intersize / len(traj_act)
precision = intersize / len(traj_rec)
F1 = 2 * precision * recall / (precision + recall)
return F1
```
**traj_act** is the expectation and **traj_rec** is the prediction. **traj_rec** is in the following format in prior studies: (trip origin, .. prediction .., trip destination). Assume that the trip query is **(Central Park, Carnegie Hall, 3)** where **Central Park** is the trip origin, **Carnegie Hall** is the trip destination, and **3** is the trip length (including trip origin and destination). We only make one (**3 - 2**) prediction as the trip origin and destination are given. Assume the expected answer is **[Central Park Zoo]** and the prediction is **[Flatiron Building]**. Prior work first wrapped both of them with the trip origin and the trip destination and sent it to **calc_F1** to compute the predictive performance. Thus, **traj_act** becomes **[Central Park, Central Park Zoo, Carnegie Hall]** and **traj_rec** becomes **[Central Park, Flatiron Building, Carnegie Hall]**. This will yield a f1 of 0.66, which is incorrect as it should be 0. The following code is an example showing that this metric will yield an incorrect result. Run this code online at [https://www.online-python.com/t215p0JNaM](https://www.online-python.com/t215p0JNaM).
```
print(calc_F1(['Central Park', 'Central Park Zoo', 'Carnegie Hall'], ['Central Park', 'Flatiron Building', 'Carnegie Hall']))
0.6666666666666666
```
## Issues with C-ILP
Another issue in C-ILP is that C-ILP peek the test sequence.

Please see the code in lines 57-60 at [https://github.com/Estrid0112/context_aware_trip_recommendation/blob/master/src/main/java/Recommender.java](https://github.com/Estrid0112/context_aware_trip_recommendation/blob/master/src/main/java/Recommender.java)

```
57 double timeBudget = Math.ceil(this.environment.getTimeCost(testSequence));
58
59 System.out.println("Time budget: " + timeBudget);
60 environment.setConstraints(testVisits, timeBudget);
```

The constraint **timeBudget** is obtained in line 57, which calls the function in lines 76-85 at [https://github.com/Estrid0112/context_aware_trip_recommendation/blob/master/src/main/java/OrienteeringEnvironment.java](https://github.com/Estrid0112/context_aware_trip_recommendation/blob/master/src/main/java/OrienteeringEnvironment.java)

```
76 private double getTimeCost(Visit[] sequence){
77 	double totalCost = 0;
78	for(int i = 0; i<sequence.length-1;i++){
79		totalCost = totalCost + getTransitCost(sequence[i].POI, sequence[i+1].POI);
80	}
81	for(int i =1; i<sequence.length-1; i++){
82		totalCost = totalCost + this.POIGraph[sequence[i].POI].visitTime;
83	}
84	return totalCost;
85 }
```

In lines 78-80 and lines 81-83, the time budget is calculated by summing up the transit time and the visiting time point by point <img src="https://render.githubusercontent.com/render/math?math=tr(p_{1}) %2b tr(p_{2}) %2b \dots %2b tr(p_{n})"> where <img src="https://render.githubusercontent.com/render/math?math=p_{1}, p_{2}, \dots, p_{n}"> is the POIs in the test sequence and <img src="https://render.githubusercontent.com/render/math?math=tr"> is the time cost function defined in [2] (Section IV).  We argue that the whole test query sequence except <img src="https://render.githubusercontent.com/render/math?math=p_{1}"> and <img src="https://render.githubusercontent.com/render/math?math=p_{n}"> must be unseen as <img src="https://render.githubusercontent.com/render/math?math=p_{2}, \dots, p_{n-1}"> are the query answer. Getting the visit duration and transit costs from the test sequence gives tighter constraints for ILP to solve and thus leads to performance gains. Thus, we made a change to the code so that the time budget was the time difference straight between the departure of <img src="https://render.githubusercontent.com/render/math?math=p_{1}"> and the arrival of <img src="https://render.githubusercontent.com/render/math?math=p_{n}">. Please see the following code change if you would like to reproduce the same results.
```
76  private double getTimeCost(Visit[] sequence){
77    return new TimeConverter().getDateDiff(testVisits[0].departureTime,testVisits[testVisits.length - 1].arrivalTime);
78  }
```
## Tutorials

### Run BERT-Trip
```
conda create --name trip_pytorch python=3.9.12
conda activate trip_pytorch
pip install -r requirements.txt
python3 run.py
```

### Run baselines
Please see files in folder /baseline/

## Reference
[1] D. Chen, C. S. Ong, and L. Xie, “Learning points and routes to recommend trajectories,” in  Proceedings of the 25th ACM International on Conference on Information and Knowledge Management, ser. CIKM ’16.  New York, NY, USA: ACM, 2016, p. 2227–2232. [Online]. Available: https://doi.org/10.1145/2983323.2983672

[2] J. He, J. Qi, and K. Ramamohanarao,  “A joint context-aware embedding for trip recommendations,” in 35th IEEE International Conference on Data Engineering, ICDE 2019, Macao, China, April  8-11, 2019. IEEE, 2019, pp. 292–303. [Online]. Available: https://doi.org/10.1109/ICDE.2019.00034

[3] K. H. Lim, J. Chan, C. Leckie, and S. Karunasekera. Personalized tour recommendation based on user interests and points  of interest visit durations. In  IJCAI, pages 1778–1784, 2015.

[4] Q. Gao, W. Wang, K. Zhang, X. Yang, and C. Miao, “Self-supervised representation learning for trip recommendation,” CoRR, vol. abs/2109.00968, 2021.

[5] Q. Gao, F. Zhou, K. Zhang, F. Zhang, and G. Trajcevski, “Adversarial human trajectory learning for trip recommendation,” IEEE Transactions on Neural Networks and Learning Systems, pp. 1–13, 2021.
