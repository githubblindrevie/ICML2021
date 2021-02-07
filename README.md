# ICML2021
## For Blind Review Only
All the running details can be checked in this repo.
Simply open the **.ipynb** file, the experiment details will be displayed.

To reproduce the experiment, few python package dependencies are required.
For simplicity, we display all the running details in jupyter notebooks.
The output of each stage is clearly shown.
### requirements:
    pytoch
    tensorflow
    pandas
    numpy
    scikit-learn

**aosr_utility.py** contains the **AOSR loss function** for tensorflow and the **isolation forest based open-set sample enrichment function**

**mnist_exp_showcase.ipynb** presents the running details of AOSR on MNIST, Omniglot, MNIST-noise, and noise dataset

**cifar10_exp_showcase.ipynb** presents the running details of AOSR on cifar10, ImageNet Resize, ImageNet Crop, LSUN Resize and LSUN Crop dataset

## Neural Network Structure

### Double Moon:
**Double Moon Encoder**: None, scince the input space is $$\mathbb{R}^2$$
**Double Moon Open-set Learning Neural Network**:

```
Dense(64, activation='relu'),
Dense(64, activation='relu'),
Dense(32, activation='relu'),
Dense(32, activation='relu'),
Dense(16, activation='relu'),
Dense(16, activation='relu'),
Dense(8, activation='relu'),
Dense(8, activation='relu'),
Dense(3),
Activation(activation='softmax')
```
To Initialize:
```
optimizer='adam',
loss='sparse_categorical_crossentropy',
learning_rate=0.001
epochs=5
```
To Learn
```
optimizer='adam',
loss=pq_risk(detetor, z_q_sample, z_q_weight, z_p_X, 0.15, 2),
learning_rate=0.001 for 20 epochs
learning_rate=0.0001 for 10 epochs
```

### MNIST
**MNIST Encoder**: Plain CNN 
```
Conv2D(filters=100, kernel_size=(3, 3),activation="relu"),
.Conv2D(filters=100, kernel_size=(3, 3), activation="relu"),
Conv2D(filters=100, kernel_size=(3, 3), activation="relu"),
MaxPooling2D(pool_size=(2, 2)),
Conv2D(filters=100, kernel_size=(3, 3), activation="relu"),
Conv2D(filters=100, kernel_size=(3, 3), activation="relu"),
MaxPooling2D(pool_size=(2, 2)),
Flatten(),
Dense(500),
Dense(10),
Activation(activation='softmax')
```
To Encode:
```
optimizer='adam',
loss='sparse_categorical_crossentropy',
learning_rate=0.001
epochs=5
```
**MNIST Open-set Learning Neural Network**:
```
Dense(11),
Activation(activation='softmax')
```
To Initialize:
```
optimizer='adam',
loss='sparse_categorical_crossentropy',
learning_rate=0.001
epochs=5
```
To Learn
```
optimizer='adam',
loss=pq_risk(detetor, z_q_sample, z_q_weight, z_p_X, 0.15, 2),
learning_rate=0.001
epochs=25
```
### CIFAR10
**CIFAR10 Encoder**: ResNet18
To Encode:
```
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
```
**CIFAR10 Open-set Learning Neural Network**:
```
Dense(11),
Activation(activation='softmax')
```
To Initialize:
```
optimizer='adam',
loss='sparse_categorical_crossentropy',
learning_rate=0.001
epochs=5
```
To Learn
```
optimizer='adam',
loss=pq_risk(detetor, z_q_sample, z_q_weight, z_p_X, 0.15, 2),
learning_rate=0.001
epochs=30
```