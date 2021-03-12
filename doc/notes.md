# Ideal State Space for RL
Context __x__ Presented Datapoint __=>__ Q
#### Context
embedding(_U_) __x__ embedding(_L_)
#### Presented Datapoint
embedding(_X_) __x__ AL features
#### Embedding
Use a pretrained ResNet18 backbone to embedd the images. </br>
Use the same backbone for the classification model and only active learn the last layer.



# Lifting problem to multiple datasets
#### Information that is considered:
Classic AL: labeled pool of single target dataset (T+) </br>
Semi-supervised: labeled and unlabeleled pool of single target dataset (T+-) </br>
Transfer learning: labeled and unlabeled pool of many training dataset + testing on an unknown dataset 

|    | BvsSB/Entropy | RL |
|---|---|---|
| Single Dataset T+ |  done | new |
| semi-supervised T+/T- | done | new |
| transfer learning T+-/other+-  | done | new |

# Additional Ideas
#### meta dataset
_image configutation_ __=>__ _classifier configuration_

# Frameworks
|    | Current | Dataset2Vec | Attention/Transformer
|---|---|---| --- |
| Data Preprocessing |  - |  |
| State | ids |  |
| Value-Function | Old-School Q(s,a) / v(s)
| Agent | Dense |  |