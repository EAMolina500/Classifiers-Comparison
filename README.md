High-Dimensional Model Simulation Tool
Overview
This repository contains a single R script that implements a simulation tool designed to evaluate the performance of classification models in high-dimensional settings. The tool is developed as part of the undergraduate thesis titled "Multinomial Logistic Regression in High Dimensions".

Purpose
The core problem addressed by this project is the evaluation of classification algorithms when the number of predictors (features) is large compared to the sample size. High-dimensional datasets pose challenges to traditional methods, making it critical to assess their accuracy, robustness, and parameter recovery capabilities.

Features
The script provides:

Synthetic Data Generation: Produces training and testing datasets with user-defined properties, including the number of features, sample size, and covariance structure.
Model Evaluation: Implements multiple classification models:
Linear Discriminant Analysis (LDA)
Multinomial Logistic Regression (MLR)
Penalized Regression models (LASSO and Elastic Net)
Random Forest
Metrics Calculation: Quantifies model performance using misclassification rates and parameter recovery metrics, such as precision and recall for penalized regression models.
Simulation Control: Allows configuration of parameters for the simulation, including the number of iterations and dataset dimensions.
Usage
To use the script, clone this repository and execute the Simulate function in R. Below is an example of how to set up and run a simulation:

# Load the script
source("Simulate.R")

# Define parameters
sigma <- diag(10)  # Covariance matrix
beta.list <- list(c(1, 0, 0, 0, 0, 0, 0, 0, 0, 0))  # True coefficients
p <- 10            # Number of predictors
n.train <- 150     # Training sample size
n.test <- 2000     # Testing sample size
R <- 50            # Number of repetitions

# Run the simulation
results <- Simulate(sigma, beta.list, p, n.train, n.test, R)

# View the results
print(results)

Output
The script generates:

Misclassification Rates: Mean and standard deviation for all models.
Precision and Recall: For LASSO and Elastic Net models.
Simulation Metadata: Total time, number of attempts, and successful iterations.
Dependencies
The script requires the following R packages:

glmnet (for LASSO and Elastic Net)
randomForest (for Random Forest)
You can install these packages using:

install.packages(c("glmnet", "randomForest"))

Author
Molina, Agustin
Undergraduate thesis project at National University of Rio Cuarto.
Year
2024

License
This project is licensed under the MIT License. See the LICENSE file for details.
