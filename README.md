# Distributional Knowledge Transfer for Heterogeneous Federated Learning

This is the code for paper: **Distributional Knowledge Transfer for Heterogeneous Federated Learning**

**Abstract:** Federated learning (FL) produces an effective global model by aggregating multiple client weights trained on their private data. However, it is common that the data are not independently and identically distributed (non-IID) across different clients, which greatly degrades the performance of the global model. We observe that existing FL approaches mostly ignore the distribution information of client-side private data. Actually, the distribution information is a kind of structured knowledge about the data itself, and it also represents the mutual clustering relations of data examples. In this work, we propose a novel approach, namely Federated Distribution Knowledge Transfer (FedDKT), that alleviates heterogeneous FL by extracting and transferring the distribution knowledge from diverse data. Specifically, the server learns a lightweight generator to generate data and broadcasts it to the sampled clients, FedDKT decouples the feature representations of the generated data and transfers the distribution knowledge to assist model training. In other words, we exploit the similarity and shared parts of the generated data and local private data to improve the generalization ability of the FL global model and promote representation learning. Further, we also propose the similarity measure and attention measure strategies, which implement FedDKT by capturing the correlations and key dependencies among data examples, respectively. The comprehensive experiments demonstrate that FedDKT significantly improves the performance and convergence rate of the FL global model, especially when the data are extremely non-IID. In addition, FedDKT is also effective when the data are identically distributed, which fully illustrates the generalization and effectiveness of the distribution knowledge.



### Dependencies

- python 3.7.9 (Anaconda)
- PyTorch 1.7.0
- torchvision 0.8.1
- CUDA 11.2
- cuDNN 8.0.4



### Dataset

- CIFAR-10
- CIFAR-100
- ImageNet-LT



### Parameters

The following arguments to the `./options.py` file control the important parameters of the experiment.

| Argument                    | Description                                       |
| --------------------------- | ------------------------------------------------- |
| `num_classes`               | Number of classes                                 |
| `num_clients`               | Number of all clients.                            |
| `num_online_clients`        | Number of participating local clients.            |
| `num_rounds`                | Number of communication rounds.                   |
| `num_epochs_local_training` | Number of local epochs.                           |
| `batch_size_local_training` | Batch size of local training.                     |
| `match_epoch`               | Number of optimizing federated features.          |
| `crt_epoch`                 | Number of re-training classifier.                 |
| `ipc`                       | Number of federated features per class.           |
| `lr_local_training`         | Learning rate of client updating.                 |
| `lr_feature`                | Learning rate of federated features optimization. |
| `lr_net`                    | Learning rate of classifier re-training           |
| `non_iid_alpha`             | Control the degree of heterogeneity.              |
| `imb_factor`                | Control the degree of imbalance.                  |



### Usage

Here is an example to run FedDKT on CIFAR-10:

```python
python main.py --num_classrs=10 \ 
--num_clients=20 \
--num_online_clients=8 \
--num_rounds=200 \
--num_epochs_local_training=10 \
--batch_size_local_training=32 \
--match_epoch=100 \
--ctr_epoch=300 \
--ipc=100 \
--lr_local_training=0.1 \
--lr_feature=0.1 \
--lr_net=0.01 \
--non-iid_alpha=0.5 \
--imb_factor=0.01 \ 
```
