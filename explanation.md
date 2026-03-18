# Handwritten Digit Recognition System: Mathematical Formulation & Architecture Design

## 1. Problem Overview
The goal is to automatically classify grayscale images of handwritten digits ($28 \times 28$ pixels) into one of 10 categories (digits 0-9) using a Convolutional Neural Network (CNN). This replicates postal code recognition systems where automatic, robust, and fast classification is required.

## 2. Model Architecture Design

The model is designed using an alternating sequence of convolutional and pooling layers, followed by dense layers mapping extracted features to target classes.

### Layer 1: Convolution Layer 1 (32 filters, $3 \times 3$)
- **Purpose**: Extracts low-level features such as lines, edges, and simple curves.
- **Input**: $28 \times 28 \times 1$
- **Mathematical Operation**: A $3 \times 3$ kernel convolves over the input matrix, computing spatial dot products. 
  Let $x$ be the input and $w$ the filter weights, the feature map $Z$ is calculated as:
  $$Z_{i,j} = \sum_{k=1}^3 \sum_{l=1}^3 x_{i+k-1, j+l-1} \cdot w_{k,l} + b$$
- **Activation (ReLU)**: Adds non-linearity $A = \max(0, Z)$. Replaces negative outputs with 0.
- **Output Shape**: $26 \times 26 \times 32$ (No padding applied, resulting in a slightly smaller output spatial dimension).

### Layer 2: Max Pooling Layer 1 ($2 \times 2$)
- **Purpose**: Reduces spatial dimensions, decreasing computational load and acting as a form of translational invariance.
- **Operation**: Extracts the maximum value from every non-overlapping $2 \times 2$ patch.
  $$P_{i,j} = \max(A_{2i:2i+2, 2j:2j+2})$$
- **Output Shape**: $13 \times 13 \times 32$.

### Layer 3: Convolution Layer 2 (64 filters, $3 \times 3$)
- **Purpose**: Leverages previously extracted edges to identify more complex abstract patterns, shapes, and corners.
- **Activation (ReLU)**: Same as Layer 1.
- **Output Shape**: $11 \times 11 \times 64$.

### Layer 4: Max Pooling Layer 2 ($2 \times 2$)
- **Purpose**: Further downsamples the feature maps.
- **Output Shape**: $5 \times 5 \times 64$.

### Layer 5: Flatten Layer
- **Purpose**: Converts the 3D tensor to a 1D vector so it can be passed into Dense layers.
- **Mathematical Operation**: Re-aligns a $5 \times 5 \times 64$ tensor into a flat array of size 1600.

### Layer 6: Fully Connected (Dense) Layer (64 units)
- **Purpose**: Performs high-level numerical reasoning based on extracted features.
- **Operation**: Computes $Z = W \cdot A + b$, where $A$ is the 1600-dimensional vector. Followed by a ReLU activation.
- **L2 Regularization**: Adds a penalty to the loss proportional to the sum of the squared weights (L2 norm), which helps prevent overfitting by discouraging overly complex models and large weights.

### Layer 7: Dropout Layer (Rate: 0.5)
- **Purpose**: Regularization technique to prevent overfitting.
- **Operation**: Randomly sets 50% of the neurons' activations to zero during training, forcing the network to distribute learned representations across multiple paths instead of relying heavily on a few neurons.

### Layer 8: Output Layer (10 units)
- **Purpose**: Predicts the probability of the input image belonging to each of the 10 digit classes.
- **Activation (Softmax)**: 
  $$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=0}^{9} e^{z_j}}$$
  Converts logits into well-calibrated probabilities summing to 1.

## 3. Loss Function and Training
- **Categorical Crossentropy Loss**: Evaluates the difference between the true one-hot encoded label $y$ and predicted probability vector $\hat{y}$.
  $$L = -\sum_{i=1}^{10} y_i \log(\hat{y}_i)$$
  The optimizer (`adam`) minimizes this loss iteratively using backpropagation to update layer weights $w$ and biases $b$.

## 4. Regularization and Dynamic Training Control
The model uses multiple regularization techniques including **Dropout**, **L2 regularization** (applied to Dense layer weights), and **Early Stopping**. Instead of fixing epochs, training is dynamically controlled based on validation performance to ensure optimal generalization. If the validation loss fails to improve for 3 consecutive epochs, the training halts and the best weights are restored.
