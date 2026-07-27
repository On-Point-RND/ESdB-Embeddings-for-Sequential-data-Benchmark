# ESdB: Embeddings for Sequential Data Benchmark

Welcome to ESdB, the repository accompanying **"Towards a Universal Sequence Embedding: A Multi-Domain, Multi-Task Benchmark and Evaluation of Self-Supervised Methods."**

ESdB is a benchmark for evaluating self-supervised sequence embeddings across multiple downstream tasks and data domains. It brings 12 source datasets into a common event-sequence representation and provides targets for classification, regression, forecasting, and anomaly detection.

## Dataset

The processed ESdB dataset is available on Hugging Face:

- [ESdB: Embeddings for Sequential Data Benchmark](https://huggingface.co/datasets/On-Point-Rnd/ESdB-Embeddings-for-Sequential-data-Benchmark)

The original data sources are listed below.

| Dataset | Domain | Source |
|---|---|---|
| AGE | Transactions | [Age Dataset](https://ods.ai/competitions/sberbank-sirius-lesson/data) |
| AlphaBattle | Transactions | [AlphaBattle 2.0](https://www.kaggle.com/datasets/mrmorj/alfabattle-20) |
| Retail | Transactions | [X5 RetailHero](https://ods.ai/competitions/x5-retailhero-uplift-modeling/data) |
| Taobao | Recommender systems | [User Behavior Data from Taobao](https://tianchi.aliyun.com/dataset/46) |
| Zvuk | Recommender systems | [Zvuk Dataset](https://www.kaggle.com/datasets/alexxl/zvuk-dataset) |
| 30Music | Recommender systems | [30Music Listening and Playlists Dataset](https://remaplab.deib.polimi.it/resources/) |
| Yambda | Recommender systems | [Yambda](https://huggingface.co/datasets/yandex/yambda) |
| Electric Devices | Time series | [ElectricDevices](https://www.timeseriesclassification.com/description.php?Dataset=ElectricDevices) |
| ETT | Time series | [Electricity Transformer Temperature Dataset](https://github.com/zhouhaoyi/ETDataset) |
| Favorita | Time series | [Corporacion Favorita Grocery Sales Forecasting](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data) |
| Rossmann | Time series | [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales/data) |
| Twitter | Text | [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140/data) |

## Preprocessing

See the [preprocessing guide](preprocess/with_shifts/README.md) for the dataset preparation and target conventions used by ESdB.
