# Top-k DDM: A novel two-stage concept drift detection framework for recommendation systems

This framework is specifically designed to address the concept drift problem in recommender systems, particularly the challenge posed by the dynamic changes in user preferences over time, aiming to improve the long-term performance of recommender systems. Traditional drift detection methods (such as DDM) rely on the error rate of classification tasks and cannot effectively capture the decline in Top-k ranking quality in recommender systems. Top-k DDM solves this problem through an innovative two-stage mechanism, making it more sensitive and robust to recommender tasks.

## Project core functions

- **drift detection method**：The core algorithm Top-k DDM was implemented.
- **Baseline algorithm integration**：It includes several classic drift detection methods as baseline comparisons.
- **Synthetic data generation**：Supports generating recommendation system datasets with artificial drift events.
- **Real dataset processing**：Supports processing and experimentation with real Twitch datasets.
- **Performance evaluation**：Calculate evaluation metrics for recommender systems such as HR@k and NDCG@k.

## Project Structure

```
Master/
├── dataset/           # Dataset processing related code
│   ├── Dataset/       # Processed dataset storage directory
│   ├── Original Dataset/ # Original dataset storage directory
│   └── data_process.py  # Data processing script
├── utils/             # Core functional modules
│   ├── framework.py   # Experimental framework
│   ├── evaluate_metrics.py # Evaluation indicators
│   ├── other_drifter.py    # Other drift detection algorithms
│   ├── synth_data.py       # Synthetic data generation
│   └── Topk_DDM.py             # Top-k DDM implementation
├── result/            # Experimental Results and Visualization
│   ├── figure/        # Experimental results figure
│   ├── table/         # Experimental Results Table
│   └── get_*.py       # Experimental script
├── main.py            # test file
└── README.md          # Project Description Document
```

## Core Component Description

Top-k DDM is our core method, which overcomes the limitation of traditional methods being unsuitable for recommendation ranking tasks. Its workflow consists of two stages:

- **Phase 1: Performance Monitoring**：This stage transforms the actual rank of the recommended item into a probabilistic error signal using a sigmoid function. The lower the rank, the higher the probability of generating the error signal.
- **Phase 2: Drift Verification**：The Classifier Two-Sample Test (C2ST) was used to determine whether there were significant differences in the distribution of the old and new data.

## Install dependencies

The project's main dependencies include:
- Python==3.12.9
- numpy==2.3.4
- pandas==2.3.3
- scikit-learn==1.7.2
- lightgbm==4.6.0
- cornac==2.3.5
- river==0.22.0
- matplotlib==3.10.7
- torch==2.9.0+cu126

## Running Example

You can modify `main.py` as a test file to test the effectiveness of different drift detection methods. This file uses two synthetic datasets: