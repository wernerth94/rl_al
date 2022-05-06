## State Spaces
--------------------------------------------------- **Internal State** ------------------------------------------------- **Sample-Dependent** ----------

| Paper                                             | Domain                                                    | Pools                                                                                      | Other      |     | Datapoint                                                                                      | Classifier                                                                                             | Prediction                              |
|---------------------------------------------------|-----------------------------------------------------------|--------------------------------------------------------------------------------------------|------------|-----|------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|-----------------------------------------|
| me                                                | Image Classfication                                       |                                                                                            | Current F1 |     | Diff of sampled point and (un)labeled pool (pools are averaged) (fixed encoder representation) |                                                                                                        | BvsSB, Entropy, <br>Histogram of output |
| [PAL](https://arxiv.org/pdf/1708.02383.pdf)       | NER                                                       | W2V + trainable CNN encoder                                                                |            |     |                                                                                                | Probability of the most probable label sequence <br> under the model (Found with a CRF + Viterbi Alg.) | Unsorted output of the model            |
| [Dreaming](https://aclanthology.org/P19-1401.pdf) | Sentiment Classification /<br> Author Profiling /<br> NER | - Sum of (un)labeled pool representations <br>- Distribution of labels in the labeled pool |            |     | Fixed encoder representation                                                             |                                                                                                        | Unsorted output of the model            |

### Other Possible Elements of the State Space
- Internal State:
  - Output of the classifier on a fixed set of held-out datapoints
  - Meta-Features of the (un)labeled set (sparsity, skewedness, etc)
- Sample Dependent:
  - Activations of the penultimate layer of the classifier
  - 

## Domains:
| Domain               | Classifier | Budget | Sample Size |
|----------------------|------------|--------|-------------|
| Image Classification | CNN        | 2000   | 20          |
| NER                  | CRF        | 200    | 5-10        |
| Author Profiling     | CNN        | 100    | 5           |
| Sentiment Class.     | CNN        | 100    | 5           |

## Further Ideas
### Reducing noise in the Environment
For each added sample, we generate augmentation and also add them to the pool <br>

### Batch Mode
Sorting the V-values and taking the Top-K. <br>
The env can take multiple actions <br>
The replay buffer receives duplicate transitions for each action a_k

### Dueling Networks
Disentangling Internal State and Sample-Dependent state <br>
Serving them as two independent inputs <br>
Dueling networks compute the advantage between internal state (context) and the sample (C51, Rainbow)

### Meta Features for the Pools
Dataset to Vec