## My deep learning and Machine Learning Workflow

### Step 1:Domian Level Knowledge

- Try to understand the overall goal of the project.
- Have a deep understanding about the problem to solve as it'll pretty much make it much easier to know what type of architecture (overall structure) to use and type of data to collect
- Understand the type of data available to work with or type of data to collect if there is no available data (Tabular data, time series, image, audio, volumetric etc.)

# Step 2: Packages Selection

- The first things is to import all the neccesary packages needed for this project (packages for data cleaning, model development, evaluation/Testing and deployment of the model)

# Step 3: Settings

- device configuration (pytorch)
- define all the hyperparameters to be used and needed to be tuned to achieve a better accuracy
- Load and explore the data

# Step 3: Define the architecture of the model such as

- Define the architecture diagramatic representation
- Define the init method

  - The number of hidden units in the input layers; which is determined by the features of the data
  - Number of total hidden layers in the model (iterative)
  - Number of hidden units in each hidden layers (iterative)
  - The output layer node units is determined by the intended outcome to achieve

- Define the forward method
  - establish connects among the layers.
  - make predictions

# step 4: Loss and optimizer definition

- Instantiate the model class
- define the specific Loss function to be used either cross entropy, MSELoss, etc
- define the optimization algorithm to be used either SGD, Adam, RMSprop, Momentum etc

# step 5: A custom function to print accuracy

# step 6: Training process

- Iterate through the data
- Reset all the gradient to zero
- Forward prop (making prediction i.e summation of all the layers (weight matrices and bias vector with their activation functions to compute the prediction)
- Compute the cost function J (difference in the real output and the predicted output
- Back prop to compute the gradient wrt J (objective function)
- Updates the model parameters [weights and bias] (optimizer step)
- Logging (print the metrics)

# step 6: Evaluation/Testing

- print the metrics
