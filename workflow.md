## My deep learning workflow

# step 1: package selection

- The first things is to import all the neccesary packages needed for this project (packages for data cleaning, model development and evaluation)

# step 2: Settings

- device configuration (pytorch)
- define all the hyperparameters to be used and need to be tuned to achieve a better accuracy
- Load and explore the data

# step 3: Define the arcitecture of the model such as

- Define the init method

  - The number of hidden units in the input layers; which is determined by the features of the data
  - Number of total hidden layers in the model (iterative)
  - Number of hidden units in each hidden layers (iterative)
  - The output layer node units is determined by the intended outcome to achieve

- Define the forward method
  - make predictions

# step 4: Loss and optimizer definition

- Instantiate the model class
- define the specific Loss function to be used either cross entropy, MSELoss, etc
- define the optimization algorithm to be used either SGD, Adam, RMSprop, Momentum etc

# step 5: Training process

- Iterate through the data
- Reset all the gradient to zero
- Forward prop (making prediction i.e summation of all the layers (weight matrices and bias vector with their activation functions to compute the prediction)
- Compute the cost function J (difference in the real output and the predicted output
- Back prop to compute the gradient wrt J (objective function)
- Updates the model parameters [weights and bias] (optimizer step)

# step 6: Evaluation/Testing

- print the metrics
