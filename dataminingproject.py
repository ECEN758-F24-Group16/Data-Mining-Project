# -*- coding: utf-8 -*-
"""DataMiningProject.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/16CEYi4v2awJ3N_SFYuJGPS18_nTUoM2F
"""

"""
Variable to decide whether to train the model or import pretrained weights
True: Retrain Model
False: Import Pretrained Model and only Test
"""
Train_Model = False

# Import Dependencies
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision import models
import torch.nn as nn
from torch.utils.data import Subset
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import shutil
import cv2
import os
from collections import Counter
import matplotlib.pyplot as plt
from IPython.display import Image
from imutils import paths
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from operator import itemgetter
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
!pip install PyGithub
from github import Github
import requests
from pathlib import Path

"""# **Data Preparation**"""

# Download Dataset (Remove Extraneous 102nd Category 'BACKGROUND_Google')
caltech101_data = torchvision.datasets.Caltech101('/content/', download=True)
shutil.rmtree('/content/caltech101/101_ObjectCategories/BACKGROUND_Google')

# Create Train/Test Split and Data Augmentation (Train/Validation Split Handled in training with Cross-Validation)

folder_path = '/content/caltech101/101_ObjectCategories'

initial_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
    transforms.Resize((250, 250)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
dataset = datasets.ImageFolder(root=folder_path, transform=initial_transform)
train_size = int(len(dataset)*0.7)
train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset)-train_size])

#  Print the labels
train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
print("Train labels:", train_labels)
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

"""# **Exploratory Data Analysis**"""

all_labels = [label for _, label in dataset]
label_frequency = Counter(all_labels)
labels_list = list(idx_to_class[label] for label in label_frequency.keys())
frequencies_list = list(label_frequency.values())
# Plot the data distribution
plt.figure(figsize=(15, 6))
plt.bar(labels_list, frequencies_list, color='blue', edgecolor='black')
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.title('Label Frequency Histogram')
plt.xticks(labels_list, rotation=90)
plt.show()

sum = torch.zeros(3)
squared_sum = torch.zeros(3)
num_batches = 0

# to calculate mean and standard deviation
for images in dataset:
    sum += images[0].sum(dim=[0, 1, 2])

    squared_sum += (images[0] ** 2).sum(dim=[0, 1, 2])
    num_batches += images[0].size(0) * images[0].size(1) * images[0].size(2)

mean = sum / num_batches
std = (squared_sum / num_batches - mean ** 2).sqrt()

print("Calculated mean for caltech101 data:", mean)
print("Calculated std for caltech101 data:", std)

image_paths = paths.list_images('/content/caltech101/101_ObjectCategories')
data, labels = [], []
fixed_size = (224, 224)
# Resize the Images
for path in tqdm(image_paths):
    label = os.path.basename(os.path.dirname(path))
    if label == "BACKGROUND_Google":
        continue
    image = cv2.imread(path)
    resized_image = cv2.resize(image, fixed_size)
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    data.append(rgb_image)
    labels.append(label)

data, labels = np.array(data), np.array(labels)

lb = LabelEncoder()
# Using label encoder - categorical to numerical
labels = lb.fit_transform(labels)
print(f"Total Number of Classes: {len(lb.classes_)}")
lb.classes_

# helper functions to load and process images
def load_images_from_category(category_name):
    category_dir = os.path.join("/content/caltech101/101_ObjectCategories", category_name)
    image_paths = [os.path.join(category_dir, filename) for filename in os.listdir(category_dir)]
    return image_paths

def process_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    resized_image = cv2.resize(image, (300, 200), interpolation=cv2.INTER_CUBIC)
    return resized_image

# Sample image
sea_horse_imgs = load_images_from_category('sea_horse')
Image(sea_horse_imgs[0])

# Sample image shape
sea_horse_test_img = process_image(sea_horse_imgs[0])
sea_horse_test_img.shape

# get the count of class - based on number of images
def get_image_count_by_category():
    categories_dir = os.path.join("/content/caltech101/101_ObjectCategories")

    categories = os.listdir(categories_dir)
    image_count_per_category = {}

    for category in categories:
        category_path = os.path.join(categories_dir, category)
        image_count_per_category[category] = len(os.listdir(category_path))

    sorted_image_count = sorted(image_count_per_category.items(), key=itemgetter(1), reverse=True)
    return sorted_image_count

data = get_image_count_by_category()
categories, counts = zip(*data)

# Plot the data distribution
plt.figure(figsize=(12, 8))
plt.barh(categories[:20], counts[:20], color='skyblue')
plt.xlabel("Number of Images")
plt.title("Top 20 Object Categories by Image Count")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# To fetch the top 5 frequent images samples
def plot_sample_images(filtered_categories, num_images=5):
    fig, axes = plt.subplots(len(filtered_categories), num_images, figsize=(15, len(filtered_categories) * 3))
    for i, category in enumerate(filtered_categories):
        if category in ["BACKGROUND_Google", "Faces_easy"]:
            continue
        category_dir = os.path.join("/content/caltech101/101_ObjectCategories", category)
        image_files = os.listdir(category_dir)[:num_images]
        for j, image_file in enumerate(image_files):
            img_path = os.path.join(category_dir, image_file)
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i, j].imshow(img_rgb)
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_title(category, fontsize=12)
    plt.tight_layout()
    plt.show()

filtered_categories = [cat for cat in categories if cat not in ["BACKGROUND_Google", "Faces_easy"]]

plot_sample_images(filtered_categories[:5])

aspect_ratios = []
image_sizes = []
# Finding the aspect ratio distribution
for category in categories:
    category_dir = os.path.join("/content/caltech101/101_ObjectCategories", category)
    for img_name in os.listdir(category_dir):
        img_path = os.path.join(category_dir, img_name)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        aspect_ratios.append(w / h)
        image_sizes.append((w, h))

plt.figure(figsize=(10, 6))
sns.histplot(aspect_ratios, bins=30, kde=True, color='green')
plt.xlabel("Aspect Ratio (Width/Height)")
plt.ylabel("Frequency")
plt.title("Image Aspect Ratio Distribution")
plt.show()

# Plot the image size distribution
widths, heights = zip(*image_sizes)
plt.figure(figsize=(10, 6))
plt.scatter(widths, heights, alpha=0.5, color='teal')
plt.xlabel("Width")
plt.ylabel("Height")
plt.title("Image Size Distribution")
plt.show()

# Number of images per category
plt.figure(figsize=(12, 8))
plt.hist(counts, bins=30, color='coral', alpha=0.7)
plt.xlabel("Number of Images per Category")
plt.ylabel("Frequency of Categories")
plt.title("Class Imbalance Analysis")
plt.show()

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from mpl_toolkits.mplot3d import Axes3D  # Import required for 3D plotting
# PCA
def preprocess_images_for_pca(data_directory, categories, img_size=(64, 64), n_components=50):
    images_flattened = []
    labels = []

    for category in categories:
        if category in ["BACKGROUND_Google", "Faces_easy"]:
            continue
        category_dir = os.path.join(data_directory, "101_ObjectCategories", category)
        for img_name in os.listdir(category_dir):
            img_path = os.path.join(category_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            images_flattened.append(img.flatten())
            labels.append(category)

    images_flattened = np.array(images_flattened)
    labels = np.array(labels)

    scaler = StandardScaler()
    images_normalized = scaler.fit_transform(images_flattened)

    pca = PCA(n_components=n_components)
    images_pca = pca.fit_transform(images_normalized)

    return images_pca, labels, images_flattened

data_directory = "/content/caltech101"
filtered_categories = [cat for cat in os.listdir(data_directory + "/101_ObjectCategories") if cat in ["Faces", "Motorbikes", "Leopards"]]

images_pca, labels, flattened_images = preprocess_images_for_pca(data_directory, filtered_categories, n_components=50)

# Create side-by-side subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# 2D PCA Projection
axes[0].set_title("2D PCA Projection of Image Dataset")
for label in np.unique(labels):
    idx = labels == label
    axes[0].scatter(images_pca[idx, 0], images_pca[idx, 1], label=label, alpha=0.5)
axes[0].set_xlabel("PCA Component 1")
axes[0].set_ylabel("PCA Component 2")
# axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 3D PCA Projection
ax_3d = fig.add_subplot(122, projection='3d')  # Create 3D plot in the second subplot
ax_3d.set_title("3D PCA Projection of Image Dataset")
for label in np.unique(labels):
    idx = labels == label
    ax_3d.scatter(images_pca[idx, 0], images_pca[idx, 1], images_pca[idx, 2], label=label, alpha=0.5)
ax_3d.set_xlabel("PCA Component 1")
ax_3d.set_ylabel("PCA Component 2")
ax_3d.set_zlabel("PCA Component 3")
ax_3d.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

mean_vector = np.mean(flattened_images, axis=0)
standardized_images = flattened_images - mean_vector

covariance_matrix = np.cov(standardized_images, rowvar=False)

print("Covariance Matrix Shape:", covariance_matrix.shape)

from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Identify the top 3 categories by frequency
label_counts = Counter(labels)
top_categories = [category for category, _ in label_counts.most_common(3)]

# Filter images and labels for the top 3 categories
top_indices = [i for i, label in enumerate(labels) if label in top_categories]
top_images = np.array(flattened_images)[top_indices]
top_labels = np.array(labels)[top_indices]

# Encode the labels for visualization
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(top_labels)

# Apply t-SNE on the filtered images
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
data_2d = tsne.fit_transform(top_images)

# Plot the t-SNE visualization
plt.figure(figsize=(10, 8))
scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels_encoded, cmap='tab10', s=10, alpha=0.8)
plt.colorbar(scatter, ticks=range(len(top_categories)), label="Categories")
plt.clim(-0.5, len(top_categories) - 0.5)
plt.title("t-SNE Visualization of Top 3 Image Categories")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")

# Add legend
handles, _ = scatter.legend_elements()
legend_labels = label_encoder.inverse_transform(range(len(top_categories)))
plt.legend(handles, legend_labels, loc="upper right", title="Categories")
plt.show()

"""# **Model**"""

# New code


num_classes = len(dataset.classes)

def generate_baseline_model():
    # ResNet18 as the baseline model
    baseline_model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
    baseline_model.fc = nn.Sequential(
      nn.Dropout(p=0.5),
      nn.Linear(baseline_model.fc.in_features, num_classes)
    )

    return baseline_model

def train_model(model, train_loader, device, optimizer, criterion):
    model.train()
    acc, total = 0, 0
    running_loss = 0.0

    # train one epoch
    for batch_inputs, batch_labels in train_loader:
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()

        # forward pass
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels)

        # backpropagation
        loss.backward()
        optimizer.step()


        running_loss += loss.item() * batch_inputs.size(0)
        pred = outputs.max(1)[1]
        acc += (pred==batch_labels).sum().item()
        total += batch_labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = acc * 100 / total
    return epoch_loss, epoch_acc

def validate_model(model, val_loader, device, criterion):
    model.eval()
    acc, total = 0, 0
    running_loss = 0.0

    with torch.no_grad():
        for batch_inputs, batch_labels in val_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            # forward pass
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)

            running_loss += loss.item() * batch_inputs.size(0)
            pred = outputs.max(1)[1]
            acc += (pred==batch_labels).sum().item()
            total += batch_labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = acc * 100 / total
    return epoch_loss, epoch_acc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
learning_rate = 0.001
momentum = 0.9
num_epochs = 5
batch_size = 32
# K-Fold Cross Validation
k = 5
k_fold = KFold(n_splits=k, shuffle=True, random_state=3)
criterion = nn.CrossEntropyLoss()

max_val_acc = 0

if Train_Model == True:
  for fold, (train_index, val_index) in enumerate(k_fold.split(train_dataset)):
      print(f"K-Fold: {fold + 1}")

      model = generate_baseline_model().to(device)

      # Split the dataset into training and validation sets
      train_subset = Subset(train_dataset, train_index)
      val_subset = Subset(train_dataset, val_index)
      train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
      val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

      optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

      for epoch in range(num_epochs):
          train_loss, train_acc = train_model(model, train_loader, device, optimizer, criterion)
          val_loss, val_acc = validate_model(model, val_loader, device, criterion)

          print(f"Epoch: {epoch+1}, "
                f"Train Loss: {train_loss:.2f}, Train Accuracy: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.2f}, Val Accuracy: {val_acc:.2f}%")

          # save the model
          if val_acc > max_val_acc:
              max_val_acc = val_acc
              torch.save(model.state_dict(), f"/content/resnet18.pth")

def generate_model_instance(model_name):
    if model_name == "resnet18":
        model = generate_baseline_model()
    elif model_name == "resnet50":
        model = models.resnet50()
    else:
        print("Model name not found")
    return model


def load_model(model_name, checkpoint_path):
    model = generate_model_instance(model_name).to(device)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint)
    return model

def main():
    model_name = "resnet18"
    checkpoint_path = "/content/resnet18.pth"
    if Train_Model == False:
      g = Github()
      asset = g.get_repo('DurwardCator/ECEN_758_Final_Project').get_latest_release().get_assets()[0]
      session = requests.Session()
      response = session.get(asset.browser_download_url, stream = True)
      dest = Path() / asset.name
      with open(dest, 'wb') as f:
          for chunk in response.iter_content(1024*1024):
              f.write(chunk)
    model = load_model(model_name, checkpoint_path)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    acc, total = 0, 0
    running_loss = 0.0

    predictions = []
    labels = []

    with torch.no_grad():
        for batch_inputs, batch_labels in test_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            # forward pass
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)

            running_loss += loss.item() * batch_inputs.size(0)
            pred = outputs.max(1)[1]

            predictions.extend(pred.cpu())
            labels.extend(batch_labels.cpu())

            acc += (pred==batch_labels).sum().item()
            total += batch_labels.size(0)

    epoch_loss = running_loss / total

    predictions = np.array(predictions)
    labels = np.array(labels)

    accuracy = accuracy_score(labels, predictions) * 100
    precision = precision_score(labels, predictions, average='weighted', zero_division=0) * 100
    recall = recall_score(labels, predictions, average='weighted', zero_division=0) * 100
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0) * 100
    conf_matrix = confusion_matrix(labels, predictions)

    print(f"Test Loss: {epoch_loss:.2f}")
    print(f"Test Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}%, Recall: {recall:.2f}%, F1-Score: {f1:.2f}%")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Predictions:")
    print(predictions)
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=labels_list)
    fig, ax = plt.subplots(figsize=(100,100))
    disp.plot(ax=ax, xticks_rotation='vertical')
    plt.show()
    return conf_matrix

conf_matrix = main()

disp = ConfusionMatrixDisplay(conf_matrix[:10,:10], display_labels=labels_list[:10])
fig, ax = plt.subplots(figsize=(10,10))
disp.plot(ax=ax, xticks_rotation='vertical')
plt.show()