## Series of important points

- The difference between a successful model and a failed model heavily lies in the choice of features used for the model. (As we can't use a wrong factor to make good decisions) Good decisions depends heavily on right factors.
- Basic Recurrent Neural Networks are affected by Vashishing and Exploding Gradients
  - Vanishing gradients are addressed by having long term dependences where LSTM and Gated Recurrent Units can solve that.
  - Exploding gradients is addressed by clipping the gradient vector
- Mean squared error and Mean Absolute Error are not suitable for deep learning models metric evaluation
- Adam optimzer is always a good choice of optimizer
- ReLU is always a good choice of activation function for hidden layers.
- Generative Adversarial Networks (GAN) Variants are suitable for image based generative modelling
- Transformer architectue variants are suitable for text and audio based generative modelling
- The seed value needs to be the same in the research and the production environment.
- Replacing the missing values with th mode value in the training set and the test set is a common practice in data science.
- We don't deploy just the machine learning or deep learning algorithm, we deploy the entire pipeline from data analysis to feature engineering to feature selection to the model.
