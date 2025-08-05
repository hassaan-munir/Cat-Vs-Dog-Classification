# Cat vs Dog Image Classification

This project is a Machine Learning solution designed to classify images as either cats or dogs using Transfer Learning with the MobileNetV2 model. It also features a web application for interactive predictions using Streamlit.

## Dataset

We used the Cats vs Dogs dataset, containing thousands of labeled images of cats and dogs.

- Dataset organized into:
  - train/ [with subfolders cats/, dogs/]
  - val/ [validation set]
  - test/ [testing set]
- All images resized to 224x224 pixels
- Normalized using ImageDataGenerator
- Augmented training images for improved generalization

## Model Architecture

The model uses Transfer Learning based on MobileNetV2 with the following layers:

- GlobalAveragePooling2D
- Dense(1024, activation='relu')
- Dense(1, activation='sigmoid')  [Binary classification output]

### Compilation

- Optimizer: Adam
- Loss: binary_crossentropy
- Metric: accuracy

## Training

- Trained for 10 epochs
- Training accuracy: 97 percent
- Validation accuracy: 99 percent
- Saved model: my_model.h5

## Web App with Streamlit

A simple Streamlit-based web app lets users upload an image and get real-time predictions.

### Features

- Upload an image of a cat or dog
- Get prediction (Cat or Dog)
- See confidence score

### Run the App

streamlit run app.py

## Requirements

Install the required libraries using:

pip install -r requirements.txt

Typical libraries used:

- tensorflow
- keras
- streamlit
- numpy
- pillow

## Live Demo

You can try the live web application here: [Live Demo Link](## Live Demo

You can try the live web application here: [Live Demo Link](https://your-link-here)


## License

This project is open source and free to use under the MIT License.

## Acknowledgments

- TensorFlow and Keras team
- Streamlit for rapid web app deployment
- Kaggle Cats vs Dogs Dataset
