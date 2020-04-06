# A cutated list of Machine learning, deep learing, Reinforcement learning Notebooks, sheet cheat, tricks, projects and code snippests.

---

[I write a daily post on ML,DL,RL Here](https://www.linkedin.com/in/olalekan-taofeek/)

---

## The 3 Approaches to defining a Neural Network / Deep learning Implementation Architecture

- Modular Approach( sutibale for both pytorch and Keras implementation where we define sequential layers)
- Manually building weight and bias (a low level approach suitable to reproduce deep learning architecture on a paper you just read or to develop a customized layer)
- class based approach (Extending module class)
- Sequential approach inside a class based approach (Mixed approach)

## This Github will entails more on;

- Machine learning
  - Supervised learning Algorithms
    - Regression
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
    - Autoencoders
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
- Name Entity Recognition
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
- Unstructured data
  - Image data
  - Volumetric data (3D)
  - Text data
  - Audio data
