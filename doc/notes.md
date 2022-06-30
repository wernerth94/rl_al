# Organizing the Information
## Domains:
| Domain           | Classifier | Budget | Sample Size |
|------------------|------------|--------|-------------|
| Image Embedding  | CNN        | 2000   | 20          |
| Cifar-10         | CNN        | 5000   | 20          |
| NER              | CRF        | 200    | 5-10        |
| Author Profiling | CNN        | 100    | 5           |
| Sentiment Class. | CNN        | 100    | 5           |


## State Spaces
|                                                   | Internal State                                            |                                                                                            |                                       |   | Sample-Dependent             |                                                                                                        |                                         |
|---------------------------------------------------|-----------------------------------------------------------|--------------------------------------------------------------------------------------------|---------------------------------------|---|------------------------------|--------------------------------------------------------------------------------------------------------|-----------------------------------------|
| **Paper**                                         | **Domain**                                                | **Pools**                                                                                  | **Other**                             |   | **Datapoint**                | **Classifier**                                                                                         | **Prediction**                          |
| me                                                | Image Classfication                                       | Sum of (un)labeled pool representations (fixed encoder)                                    | Current F1 <br> added_images / budget |   |                              |                                                                                                        | BvsSB, Entropy, <br>Histogram of output |
| [PAL](https://arxiv.org/pdf/1708.02383.pdf)       | NER                                                       | W2V + trainable CNN encoder                                                                |                                       |   |                              | Probability of the most probable label sequence <br> under the model (Found with a CRF + Viterbi Alg.) | Unsorted output of the model            |
| [Dreaming](https://aclanthology.org/P19-1401.pdf) | Sentiment Classification /<br> Author Profiling /<br> NER | - Sum of (un)labeled pool representations <br>- Distribution of labels in the labeled pool |                                       |   | Fixed encoder representation |                                                                                                        | Unsorted output of the model            |

### Other Possible Elements of the State Space
- Internal State:
  - Output of the classifier on a fixed set of held-out datapoints
  - Meta-Features of the (un)labeled set (sparsity, skewedness, etc)
- Sample Dependent:
  - Activations of the penultimate layer of the classifier
  - 

## Policies
| Paper                   | Layer Type | Scope                      | # Features | # Hidden Layers | Layersize |
|-------------------------|------------|----------------------------|------------|-----------------|-----------|
| me                      | Dense      | Instance                   | 782        | 1               | 200       |
| PAL                     | Dense      | Instance                   |            |                 |           |
| Imitation Learning      | Dense      | Instance (TimeDistributed) | 104        | 0               | -         |
| Dreaming (Author Prof.) | Dense      | Instance (TimeDistributed) | 104        | 0               | -         |
| Dreaming (NER)          | Dense      | Instance                   | 332        | 1               | 256       |

## TODO
- pondering the meaning of life

# Further Ideas
### Reducing noise in the Environment
For each added image, we generate augmentations and also add them to the pool <br>

### Batch Mode
Sorting the V-values and taking the Top-K. <br>
The env can take multiple actions <br>
The replay buffer receives duplicate transitions for each action a_k

### Dueling Networks
Disentangling Internal State and Sample-Dependent state <br>
Serving them as two independent inputs <br>
Dueling networks compute the advantage between internal state (context) and the sample (C51, Rainbow)

### Imitation Learning
Imitation Learning optimizes the likelyhood of the policy recreating the best known solutiuon (BvsSB) <br>
This possibly guides the model through the complex state stace with image embeddings

### Meta Features for the Pools
Dataset to Vec

### Model-Based RL
- s' would be averages of the sample (like in the memory buffer)
- contrastive Loss for Encoder model?