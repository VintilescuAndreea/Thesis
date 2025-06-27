import time

import pennylane as qml
from pennylane import numpy as np

import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet50_Weights

n_qubits = 4  # Number of qubits
q_depth = 10  # Depth of the quantum circuit (number of variational layers)
q_delta = 0.01  # Initial spread of random quantum weights
start_time = time.time()

dev = qml.device("default.qubit", wires=n_qubits)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def H_layer(nqubits):
    """
      Layer of single-qubit Hadamard gates.
    """
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)


def RY_layer(w):
    """
      Layer of parametrized qubit rotations around the y axis.
    """
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)


def entangling_layer(nqubits):
    """
      Layer of CNOTs followed by another shifted layer of CNOT.
    """
    # In other words it should apply something like :
    # CNOT  CNOT  CNOT  CNOT...  CNOT
    #   CNOT  CNOT  CNOT...  CNOT
    for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
        qml.CNOT(wires=[i, i + 1])


@qml.qnode(dev, interface="torch")
def quantum_net(q_input_features, q_weights_flat):
    """
    The variational quantum circuit.
    """

    # Reshape weights
    q_weights = q_weights_flat.reshape(q_depth, n_qubits)

    # Start from state |+> , unbiased w.r.t. |0> and |1>
    H_layer(n_qubits)

    # Embed features in the quantum node
    RY_layer(q_input_features)

    # Sequence of trainable variational layers
    for k in range(q_depth):
        entangling_layer(n_qubits)
        RY_layer(q_weights[k])

    # Expectation values in the Z basisT
    exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_qubits)]
    return tuple(exp_vals)


class DressedQuantumNet(nn.Module):
    """
    Torch module implementing the *dressed* quantum net.
    """

    def __init__(self, num_classes=5):
        """
        Definition of the *dressed* layout.
        """

        super().__init__()

        self.pre_net = nn.Linear(8, n_qubits)
        self.q_params = nn.Parameter(q_delta * torch.randn(q_depth * n_qubits))
        self.post_net = nn.Linear(n_qubits, num_classes)

    def forward(self, input_features):
        """
        Defining how tensors are supposed to move through the *dressed* quantum
        net.
        """

        # obtain the input features for the quantum circuit
        # by reducing the feature dimension from 512 to 4
        pre_out = self.pre_net(input_features)
        q_in = torch.tanh(pre_out) * np.pi / 2.0

        # Apply the quantum circuit to each element of the batch and append to q_out
        q_out = torch.Tensor(0, n_qubits)
        q_out = q_out.to(device)
        for elem in q_in:
            q_out_elem = torch.hstack(quantum_net(elem, self.q_params)).float().unsqueeze(0)
            q_out = torch.cat((q_out, q_out_elem))

        # return the two-dimensional prediction from the postprocessing layer
        return self.post_net(q_out)


def create_model(num_classes=5):
    model = torchvision.models.resnet50(pretrained=False, weights=ResNet50_Weights)

    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        # nn.Linear(1920,512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 128),
        nn.Linear(128, 64),
        nn.ReLU(inplace=True),
        nn.Linear(64, 16),
        # nn.Linear(32,16),
        nn.Linear(16, 8),
        DressedQuantumNet(num_classes=num_classes)
    )
    return model


from transformers import ViTForImageClassification

### alt model posibil
from transformers import SwinForImageClassification, BeitForImageClassification


def create_model_vit(num_classes=5):
    # Load the DeiT model pre-trained on ImageNet
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224',
                                                      num_labels=num_classes, ignore_mismatched_sizes=True)
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the classifier head layers
    for param in model.classifier.parameters():
        param.requires_grad = True


    return model


if __name__ == "__main__":
    model = create_model_vit()
    print(model)
    """
    # optional + QNN
    model.classifier = nn.Sequential(
        nn.Linear(384, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256,128),
        nn.Linear(128,64),
        nn.ReLU(inplace = True),
        nn.Linear(64,16),
        # nn.Linear(32,16),
        nn.Linear(16,8),
        DressedQuantumNet(num_classes=num_classes)
    )
    """