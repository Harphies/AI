# A cutated list of Machine learning, deep learing, Reinforcement learning Notebooks, sheet cheat, tricks, projects and code snippests.

---

[I write a daily post on ML,DL,RL Here](https://www.linkedin.com/in/olalekan-taofeek/)

---

## This Github will entails more on;

- Machine learning
  - Supervised learning Algorithms
    - Regression (where the output variable is a real value such as unique number, dollars, salary, weights or pressure)
      - Linear Regression
      - Logistic Regression
    - Classification
      - Binary Classifications
      - Multi-class classifications
      - Ordinal Classifications
  - Neural Networks and Deep Neural Networks (Deep Learning)
    - Multilayer perceptrons/ Feedforward Networks
    - Convolutional Neural Networks
      - Regional Convolutional Neural Networks (R-CNN)
      - Fast R-CNN
      - Faster R-CNN
    - Recurrent Neural Networks
      - Basic RNN (Unidirectional and Bidirectional)
      - LSTM (Long short term Memory) (Unidirectional and Bidirectional)
      - Gated Recurrent Unit (GRU) (Unidirectional and Bidirectional)
      - Attention mechanism
  - Unsupervised learning Algorithms
    - Clustering: Organizing unlabelled data into similar groups called cluster.The primary goal is to find similarities in the data points and group similar data points into cluster.
    - Anomaly Detection: It's the method of identifying rare items, events or observations which differ significantly from the majority of the data. We generally look for anomalies or ouliers in the data because they are suspicious. It's often used in Bank and Medical Error detection.
    - Autoencoders
      - FeedForward Autoencoder (Using Linear Layers to construct the Encoder and the Decoder components)
      - Convolutional Autoencoder (Using Convolutional Layers to construct the Encoder and Decoder or ConvolutionTranspose to construct the Decoder)
      - LSTM Autoencoder (Using LSTM to construct the Encoder and the Decoder )
    - Generative Adversarial Networks (GAN)
    - Deep Belief Networks (stacked Restricted Boltzmann machine)
  - Self Supervised learning
  - Semi-Supervised Algorithms
  - Reinforcement learning
- Recommendation Systems (Information retrieval)
  - Content Based Filtering
  - Collaborative Based Filtering
    - Item Based
    - User Based
  - Hybrid Systems
- Transfer Learning (domain adaptation)
  - Working with pretrained models

## Application Areas

- Computer Vision

  - Image classification: The input to the problem is an image and the required output is simply a prediction of the class that Image belong to.
  - Object Detection: The input to the problem is an Image and the required output are bounding boxes surrounding the detected objects.(identification of many classes in an image)
  - Advanced Object Segmentation: The input to the problem are images and the required output are pixels grouping that to each class
  - Neural Style Transfer

- Sequence Modelling

  - Natural Language Processing

    - Sentiment Analysis
    - Neural Machine Translation
    - Character level language modelling for text generatio
    - Poem generation

  - Speech Recognition

    - Trigger word detection

- Music Generation
- DNA Sequence Analysis
- Video Activity Recognition
- Name Entity Recognition (NER)
- Emojify

  ## Popular Types of Layers in DNNs

  - Fully connected Layer (Dense)
    - Feedforward Neural Networks is also called Multilayer perceptrons
  - Convolutional Layer
    - Feed forward, sparsely-connected / weight sharing
    - Convolutional Neural Network (CNN)
    - Typically used for Images
  - Recurrent Layer
    - Feedback
    - Recurrent Neural Network (RNN)
    - Typically used for time-series/sequential data (e.g speech and language)
  - Attention Layer/Mechanism
    - Attention(matrix multiply) + feed forward, fully connected
    - Transformer

# Types of data Neural Network and machine Learning work with

- Structured data
  - Tabular data (rows and column)
    - Spread sheets (csv)
  - Time series data
    - Univariate
    - Multivariate
- Unstructured data
  - Image data
  - Volumetric data (3D)
    - Computed tomography (CT) scans
  - Text data
  - Audio data

## The 3 Approaches to defining a Neural Network / Deep learning Implementation Architecture

- Class based approach (Extending module class)
- Modular Approach( sutibale for both pytorch and Keras implementation where we define sequential layers)
- Sequential approach inside a class based approach (Mixed approach)
- Manually building weight and bias (a low level approach suitable to reproduce deep learning architecture on a paper you just read or to develop a customized layer)
