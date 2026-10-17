# Toxic Comment Classification Using LSTM

## Project Overview

This project develops a deep learning model to classify online comments into five toxicity categories using Natural Language Processing (NLP) and a Long Short-Term Memory (LSTM) neural network.

The objective is to explore how deep learning can identify different forms of harmful language and support automated content moderation.

## Dataset

The project uses the Jigsaw Toxic Comment Classification dataset, containing 159,571 Wikipedia comments.

The original toxicity labels were converted into five mutually exclusive classes using a priority-based labeling approach:

- Clean
- Toxic
- Severe Toxic
- Threat
- Identity Attack

After text cleaning and duplicate removal, 158,216 comments were retained.

## Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas and NumPy
- Matplotlib and Seaborn
- NLP and LSTM

## Methodology

1. Performed data inspection, text cleaning, and duplicate removal.
2. Converted the original toxicity labels into five classification categories.
3. Applied tokenization and sequence padding (maximum length: 100).
4. Used an 80/20 stratified training-test split.
5. Applied class weighting to address class imbalance.
6. Developed and trained an LSTM-based neural network.
7. Evaluated model performance using accuracy, precision, recall, F1-score, and a confusion matrix.

## Model Architecture

The neural network consists of:

- Embedding Layer (64 dimensions)
- SpatialDropout1D (0.3)
- LSTM (64 units)
- Dense Layer (64 units, ReLU)
- Dropout (0.3)
- Output Layer (5 units, Softmax)

The model was trained using the Adam optimizer, sparse categorical cross-entropy loss, and Early Stopping.

## Results

The model achieved **92.75% test accuracy** on 31,644 test samples.

Performance was evaluated using a classification report and confusion matrix.

The results showed strong performance on the Clean category, while minority classes, particularly Threat and Identity Attack, remained challenging due to class imbalance.

## Interactive Prediction

The notebook also includes a prediction function that accepts user-entered text and returns its predicted toxicity category.

## Limitations

The dataset is highly imbalanced, and overall accuracy does not fully reflect performance across all toxicity categories. Further improvements could focus on minority-class detection and model generalization.

## How to Run

1. Download the Jigsaw Toxic Comment Classification dataset from Kaggle.
2. Place `train.csv` in the working directory.
3. Install the required Python libraries.
4. Open the Jupyter Notebook.
5. Run the cells sequentially to preprocess the data, train the model, and evaluate its performance.

## Dataset Source

[Jigsaw Toxic Comment Classification Challenge – Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
