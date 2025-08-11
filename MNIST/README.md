# MNIST Dataset
Applying Different Models to MNIST Dataset

#### Model 1: KNN (K-Nearest Neighbors) 
- Uses Manhattan as the distance metric
- Can Change the value of K to affect the number of neighbors used in classification task
- The most common class among neighbors will be selected
- Accuracy metric is used for evaluation

#### Model 2: MLP (Multi Layer Perceptron)
- Three FC layers (2 hidden layers and 1 output layer)
- ReLU activation to introduce non-linearity
- SGD as optimizer with learning rate of 0.01
 
#### Model 3 - LeNet-5 (Convolution Neural Networks)
- This implementation of the LeNet model consists of two convolutional layers followed by average pooling, three fully connected layers, and activation functions including Tanh and softmax for classification tasks.
- Original input size in LeNet-5 is 32 * 32 but this implementation has input size of 28 * 28
 
#### Model 4 - CAN (Context Aggregation Networks) Model
- Simple implementation of CNN with dilated convolutions (dilated convolutions introduces gaps in kernel elements to capture more contextual information and preserves spatial dimensions (28*28))
- Uses Leaky ReLU (compared to ReLU, instead of making it 0 when input is less than or equal to 0, it makes it alpha * input) to have a non-zero gradient when input is negative
- Uses dilated convolutions with increasing rates to capture features at various scales while preserving spatial resolution

## Results
- KNN with K = 3 achieved a 96.33% accuracy
- MLP with 256 hidden neurons achieved a 96.15% accuracy
- LeNet achieved a 93.84% accuracy
- CAN with 16 feature channels achieved a 98.58% accuracy. (more feature channels might lead to overfitting - drop in accuracy)
