## State Spaces
Paper | Internal State | Sample-Dependent

| Paper                                                                     | Domain                                         | Pools                                                          | Other      |     | Classifier                                                                                             | Prediction                                  |
|---------------------------------------------------------------------------|------------------------------------------------|----------------------------------------------------------------|------------|-----|--------------------------------------------------------------------------------------------------------|---------------------------------------------|
| me                                                                        | Image Classfication                            | Fixed Embeddings <br> Diff of sampled point with averaged pool | Current F1 |     |                                                                                                        | BvsSB, Entropy, <br>Histogram of Prediction |
| [PAL](https://arxiv.org/pdf/1708.02383.pdf)                               | NER                                            | Fixed W2V + trainable CNN                                      |            |     | Probability of the most probable label sequence <br> under the model (Found with a CRF + Viterbi Alg.) | Unsorted output of the model                |
| [Dreaming](https://aclanthology.org/P19-1401.pdf#page=9&zoom=100,402,942) | Sentiment Classification <br> Author Profiling |                                                                |            |     |                                                                                                        |                                             |

## Further Ideas
### Dueling Networks
Disentangling Internal State and Sample-Dependent state

### Meta Features for the Pools
Dataset to Vec