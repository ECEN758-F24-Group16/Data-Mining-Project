# ECEN 758 Data Mining Project

## Introduction

### Overview of dataset

We used the Caltech101 dataset for the project. The Caltech 101 dataset, introduced by Fei-Fei et al. in 2004, is a widely used dataset for object recognition tasks. It contains 9,144 im- ages spanning 101 distinct object categories, such as animals, vehicles, instruments, and everyday objects, and an additional background category. Each category contains between 40 to 800 images, with most categories having about 50 images. The images are of varying sizes and resolutions, presenting real- world challenges for object classification tasks. The dataset provides a balanced variety of classes, making it suitable for multi-class classification problems in computer vision tasks.

### Experiments
In this project, we aim to develop a classification model for the Caltech 101 dataset. The following key steps are followed:
- Data Preparation: We perform data cleansing and transformations of the images. We also split the dataset into training, validation, and test sets
- Exploratory Data Analysis (EDA): Descriptive statistics and visualizations are used to better understand the dataset, including class distributions, image sizes, and example visualizations of different classes. Dimensionality reduction techniques like PCA and t-SNE are applied to visualize patterns in the dataset.
- Architecture Selection: We utilize a ResNet-18 backbone, leveraging its proven performance in image classification tasks. We fine-tune the model for our dataset to optimize performance.
- Model Evaluation: We use performance metrics such as accuracy, precision, recall, and F1-score to evaluate the model’s effectiveness

## Method

### Data Preprocessing

We applied various pre-processing steps to the data. Since some of the images were grayscale, we converted all the images to RGB. To keep a consistent size, we resized all the images to a standard size of 250x250. We then randomly crop images to a size of 224 to make the model generalize better. We also applied RandomHorizontalFlip transform to make the model more robust and accurate. Finally, we normalize the images over the pixel values with the mean and standard deviation of ImageNet database since we will be using transfer learning. We lastly split the data into a training set consisting of 70% of the data sampled randomly and a test set comprised of the remaining 30%.

### Exploratory Data Analysis (EDA)

We analyzed the Caltech101 dataset to understand it better. The categories, airplanes and motorbikes, have the maximum number of samples, with more than 700 samples in each category. We have analyzed images for the top 5 categories by count below to see the generic differences in images. This shows that it contains images with different backgrounds and orientations.

### Model Selection

#### Algorithm selection

For this project, we chose a Convolutional Neural Network (CNN), specifically ResNet-18, as the classification model. CNNs are widely regarded as the most effective algorithms for image classification tasks due to their ability to automatically learn hierarchical features from images. This hierarchical feature extraction eliminates the need for manual feature engineering by identifying low-level features such as edges and textures, as well as high-level patterns like shapes and objects. Furthermore, CNNs exhibit spatial invariance through convolutional and pooling layers, allowing them to detect features regardless of their position within the image.

CNNs also excel in scalability and robustness, making them suitable for datasets of various sizes and complexities. ResNet-18, a pre-trained CNN architecture, leverages transfer learning to accelerate model development and enhance performance by utilizing knowledge from larger datasets such as ImageNet. Its proven success in similar image classification tasks makes it a compelling choice for the Caltech 101 dataset.

ResNet-18 was selected for its efficiency, robustness, and compatibility with the Caltech 101 dataset. A key feature of ResNet-18 is its use of residual connections, which mitigate the vanishing gradient problem by enabling gradients to flow directly through shortcut connections. This allows for effective training of deeper networks without performance degradation, a common issue in earlier CNN architectures.

The architecture’s lightweight design, consisting of 18 layers, strikes a balance between computational feasibility and model complexity. It is efficient enough to train on standard hardware while maintaining sufficient depth to learn diverse patterns present in the Caltech 101 dataset. Additionally, ResNet-18’s pre-trained weights from ImageNet facilitate transfer learning, reducing training time and improving generalization on the medium-sized dataset.

#### Model Building

We used the pre-trained ResNet-18 model and customized its fully connected (FC) layer to match the number of classes in the Caltech 101 dataset. The modified FC layer includes a dropout layer with a 50\% dropout rate to reduce overfitting and a linear layer that outputs predictions for the dataset’s 101 classes. This configuration helps ensure that the model can generalize effectively while retaining the knowledge learned from the pre-trained weights.

With this modified architecture established, we then fine-tuned the network weights over 5 epochs of the training dataset. We used the Stochastic Gradient Decent optimizer with a learning rate of 0.001, momentum of 0.9, and a batch size of 32. We employed a 5-fold cross validation strategy to select the network weights with the largest validation accuracy.

## Experimental Results and Discussion

## Conclusion