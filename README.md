# EPFL Machine Learning Project 1- Fall 2018
### Authors:
Alireza Mohammadshahi

Mohammadreza Banaei

## Structure of code

The code is structured as follows

```
project1_machine_learning
 |
 |--implementations.py
 |
 |--helpers.py
 |
 |--run.py
```

The `implementations.py` file contains all required function to train and predict labels. The `helpers.py` file contains all helper function such as load and save csv files, mode, ... .  
You can give the path of 'train.csv' and 'test.csv' to `run.py` file to get the proper result. There is a flag_model that indicates which model to run (0 for neural network, 1 for regularized logistic regression)

```Bash
python run.py
```

## API

We have implemented several APIs to ease the train and test process, some of them are as follows:

### Feature Generation

After cleaning the data and adding bias feature, you can use `build_data` function to generate the required features.

#### `build_data(class0_data,class1_data,class23_data)`

In this function we add all the necessary features for finding the best model. Also, we use "log(1+x)" for heavy tailed features.  
It generates features like invariant and transverse mass, sin, cos, norm, poly_features, .... 

### Data preprocessing

This part contains splitting the train and test datas into 3 different classes (0,1,23), finding the missing data in feature 0 and replacement, adding bias term and normalizing the features.  

It has the below functions:  
 - `row_split(yb,input_data,delete_col,kind)`: split the train data into the required class type. (class type: 0, 1, 23)  
 - `row_split_test(input_data,delete_col,kind)`: split the test data into the required class type. (class type: 0, 1, 23)
 - `standardize(x)`: It would normalize each feature for better convergence

In `build_data` functions, we also use "log(1+x)" function for heavy tailed features.  


### Model selection

There is flag named `FLAG_MODEL` that you can use it to change the model of training in the `run.py` file (0 for Neural Network model and 1 for regularized logistic regression which gives better result between six different models when we have a single neuron)  

In the first model, we use a simple neural network with one hidden layer. we trained it with the below parameters:  
 - `learning_rate`: 0.0051
 - `epoch`: 80 
 - `Ensemble`: 58

In this model we have a function named `post_process(Y_pred_temp,P_pred_temp)` that is explained in the next section.  

In the second model, we use regularized logostic regression since it has a better results with cross-valdation in the `predict_group` function with F-score criteria. So, in the `run.py` file we delete other optimization algorithms. The used parameters for training are as follows:  
 
 - `learning_rate`: 0.0004
 - `max_iters`: 2000
 - `Ensemble`: 58
 - `Lambda`: 10000  
** We use different lambda for each class for better fit**  

In this model, since we don't have any probability to use in `post_process` function, we just use `mode` function to predict the result.  

### Post-Processing
#### `post_process(prediction, probability)`

We use this function as a part of first model to predict the result better. The strategy is named `Ensemble` that we try to run the model with different initializtions, and use all these predictions to find the best model.  
As mentioned above, we have 58 `Ensemble` as the input (predictions and probabilities). In each step, the function tries to find the best models between the models that are available based on the probability of the model, `x_mode` function does this selection process.  

#### `x_mode(x)`

This function returns the mode of entries and their frequencies.

