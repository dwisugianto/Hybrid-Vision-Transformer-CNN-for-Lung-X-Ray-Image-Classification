**Hybrid Vision Transformer-CNN for Lung X-Ray Image Classification**
**Project Description**
This project implements a hybrid model combining Vision Transformer (ViT) and CNN (ResNet-18) for lung X-ray image classification. The model is designed to improve classification performance by leveraging ViT’s ability to capture global dependencies and CNN’s strength in extracting local features.

**System Requirements**
Python 3.x
PyTorch
Torchvision
timm (Library for Vision Transformer)
scikit-learn
matplotlib
seaborn
Installation
Clone the Repository (Optional)


git clone <repository-url>
cd <repository-folder>
Install Dependencies


pip install torch torchvision timm scikit-learn matplotlib seaborn
Prepare the Dataset

Ensure that the lung X-ray image dataset is available in an ImageFolder format inside a folder named Lung X-Ray Image.
The structure should look like this:

Lung X-Ray Image/
├── Class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Class2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...

**How to Run**
Train the Model
Run the main script:


python cs-113323-Source_Code_ViT-CNN.py
This will:

Check if GPU is available.
Load the dataset and split it into training & validation sets.
Train the Vision Transformer and CNN hybrid model.
Save the trained model as hybrid_vit_cnn_model.pth.

**Evaluate the Model**
After training, the model will:

Generate a classification report with precision, recall, and f1-score.
Display a confusion matrix.
Show the class distribution of the dataset.
Grad-CAM Visualization

The model will generate feature map visualizations using Grad-CAM for interpretability.
The visualization is saved as feature_map_visualization.png.

**Results & Outputs**
Trained Model: hybrid_vit_cnn_model.pth
Model Evaluation: Accuracy, precision, recall, and f1-score
Confusion Matrix
Grad-CAM Feature Map Visualization

**License**
This project is free to use for academic and research purposes.
