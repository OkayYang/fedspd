# FedSPD: Structured Prototype-based Knowledge Distillation for Federated Learning


## Overview

Federated Learning (FL) faces data heterogeneity as one of its primary challenges, resulting in slow convergence and degraded performance. Knowledge distillation methods have recently been recognized as an effective approach to address this challenge. However, existing federated distillation methods have the following issues:

1. **Proxy Data Dependency**: Most methods rely on proxy data that is usually impractical to obtain in reality
2. **Communication Overhead**: Although many methods apply generative models to synthesize proxy data, this increases communication overhead and causes uncertainty due to variable generator quality
3. **Privacy Risks**: Most methods risk label distribution leakage, potentially compromising participant privacy

To address these issues, we propose **FedSPD**, a structured prototype-based knowledge distillation framework that eliminates the need for proxy data or generative models, while also avoiding the exposure of label distribution. This method maintains a teacher model and employs dual knowledge distillation at both decision and representation levels, enabling clients to align with the teacher through structured prototype matching and achieve global knowledge learning.

Furthermore, for bandwidth-constrained environments, we introduce **FedSPD-LC**, a light communication variant that significantly reduces communication costs while outperforming most existing federated learning methods.


## Datasets
- [x] **FEMNIST Dataset**: Federated Extended MNIST dataset with 3,550 users and 80,526 samples, simulating realistic non-IID scenarios
- [x] **MNIST Dataset**: Classic handwritten digit classification dataset
- [x] **CIFAR10 Dataset**: 10-class color image classification dataset
- [x] **CIFAR100 Dataset**: 100-class color image classification dataset
- [x] **TinyImageNet Dataset**: 200-class image classification dataset


## Requirements

- Python 3.9+
- PyTorch 1.9+
- Other dependencies: `torchvision`, `numpy`, `matplotlib`, `scikit-learn`, etc. (see `requirements.txt` for details)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/OkayYang/fedspd.git
   cd fedspd
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command Line Interface

```bash
# Run FedSPD with default parameters
python main.py --strategy fedspd --dataset cifar10 --partition dirichlet --num_clients 10 --dir_beta 0.3

# Run FedSPD-LC for bandwidth-constrained environments
python main.py --strategy fedspdlc --dataset cifar10 --partition dirichlet --num_clients 10 --dir_beta 0.3

# Compare with baseline methods
python main.py --strategy fedavg --dataset cifar10 --partition dirichlet --num_clients 10 --dir_beta 0.3
```

#### Available Parameters:

- **Dataset Parameters**:
  - `--dataset`: Dataset name (femnist, mnist, cifar10, cifar100, tinyimagenet)
  - `--partition`: Data partitioning method (iid, noiid, dirichlet)
  - `--dir_beta`: Dirichlet distribution parameter for non-IID scenarios
  - `--num_clients`: Number of clients

- **Training Parameters**:
  - `--batch_size`: Training batch size
  - `--local_epochs`: Number of local training epochs
  - `--comm_rounds`: Number of communication rounds
  - `--ratio_client`: Proportion of clients participating in each round
  - `--lr`: Learning rate
  - `--optimizer`: Optimizer type (adam or sgd)

- **Algorithm Parameters**:
  - `--strategy`: Federated learning strategy (fedspd, fedspdlc, fedavg, fedprox, moon,fedgkd, feddistill, fedgen)

### Code Usage

#### 1. Load Dataset
```python
from fl.data import datasets

# Load dataset with clients
client_list, dataset_dict = datasets.load_dataset(
    dataset_name="cifar10",
    partition="dirichlet",
    num_clients=10,
    dir_beta=0.3
)
```

#### 2. Configure Model and Training Parameters
```python
from fl.model.model import CNN
from fl.client.fl_base import ModelConfig
from fl.utils import optim_wrapper
import torch.optim as optim

loss_fn = torch.nn.CrossEntropyLoss
optim_fn = optim_wrapper(optim.Adam, lr=1e-2)

model_config = ModelConfig(
    model_fn=CNN,
    loss_fn=loss_fn,
    optim_fn=optim_fn,
    epochs=10,
    batch_size=32
)
```

#### 3. Start Federated Learning with FedSPD
```python
from fl.server.fl_server import FLServer

# Initialize FedSPD server
fl_server = FLServer(
    client_list=client_list,
    strategy="fedspd",  # or "fedspdlc" for light communication variant
    model_config=model_config,
    client_dataset_dict=dataset_dict
)

# Start training
history = fl_server.fit(comm_rounds=100, ratio_client=0.8)
```

#### 4. Visualize Results
```python
from fl.utils import plot_global_metrics, plot_worker_metrics

# Plot training metrics
plot_global_metrics(history)
plot_worker_metrics(history)
```

### Experimental Comparison

The framework provides comprehensive comparison tools:

```bash
# Run comparison experiments for all algorithms
bash run_comparison.sh

# Generate comparison plots
python compare_strategies.py
```

This will generate comparison results showing:
- Training Loss curves
- Test Loss curves  
- Test Accuracy curves
- Communication efficiency analysis


## Contributing

We welcome contributions to improve FedSPD! Please feel free to submit issues and pull requests.
