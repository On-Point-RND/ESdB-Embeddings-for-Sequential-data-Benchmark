
T1

**Can one sequence embedding serve many downstream tasks? Could self-supervised methods from different domains transfer to event sequences?**  
We provide a comprehensive study of these questions through a **new unified benchmark we introduce**, comprising 12 datasets covering *transactions, recommender logs, time series*, and *text*. Seven self-supervised methods from different domains are evaluated across classification, regression, forecasting, and anomaly detection under a single protocol.

T2

Possible strategies for obtaining embeddings that work well across different downstream tasks

T3

**Success on one downstream task does not reliably transfer to another.**  
Task performance and method rankings are only weakly aligned, even within the same dataset. A representation selected for one task therefore cannot be assumed to work equally well for others.

T5

**There is no universally best sequence embeddings method.**
Methods perform differently across downstream tasks. Almost every method wins somewhere, but none performs consistently best across the benchmark.

T6

**Embedding fusion offers an alternative to a single universal method.**
We evaluate fusion strategies from simple concatenation and dimensionality reduction to more advanced approaches like Low-Rank Fusion. Across multiple downstream tasks, fused representations can match or even outperform the individual embeddings.

T7

**Aggregation depends on the target configuration.**
The last embedding is strongly preferred for forecasting and moderately preferred for regression, while mean pooling is more common for classification and anomaly detection. The same overall pattern holds across methods, with some differences in strength.

T8

**Label-free signals are noisy and do not reliably predict downstream performance.**

**SSL loss is not a reliable selection signal.**  
Across datasets and models, SSL loss provides no stable signal for either Optuna-based model selection or early stopping: its relationship with downstream performance varies widely across tasks and settings.

**Unsupervised metrics do not reliably identify the best checkpoint.**  
We evaluate RankMe, LiDAR, and aSMI as potential label-free signals across downstream tasks, but their behavior remains noisy. Different variants peak at different checkpoints and do not consistently recover the downstream-best model.

> (For this block - I would prefer a slider showing all pics from the paper (Figures 15-18))

T9

**Multi-target HPO reduces search cost without a consistent loss in quality.**  
Instead of running a separate HPO study for each downstream task, we optimize all tasks jointly within a single study. This requires 4× fewer trials on X5-Retail and 3× fewer on Twitter, while neither single- nor multi-target HPO consistently wins in downstream performance.