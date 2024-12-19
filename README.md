# High-Dimensional Model Simulation Tool Overview

### Undergraduate thesis project at the National University of Río Cuarto.

This repository contains a R script that implements a simulation tool designed to evaluate the performance of classification models in high-dimensional settings. The tool is developed as part of the undergraduate thesis titled "Multinomial Logistic Regression in High Dimensions".

# Purpose

The core problem addressed by this project is the evaluation of classification algorithms when the number of predictors (features) is large compared to the sample size. High-dimensional datasets pose challenges to traditional methods, making it critical to assess their accuracy, robustness, and parameter recovery capabilities.

# Features

### <ins>Script provides</ins>:

- <ins>Synthetic Data Generation</ins>: Produces training and testing datasets with user-defined properties, including the number of features, sample size, and covariance structure.

- <ins>Model Evaluation</ins>: Implements multiple classification models: Linear Discriminant Analysis (LDA) Multinomial Logistic Regression (MLR) Penalized Regression models (LASSO and Elastic Net) Random Forest

- <ins>Metrics Calculation</ins>: Quantifies model performance using misclassification rates and parameter recovery metrics, such as precision and recall for penalized regression models.

- <ins>Simulation Control</ins>: Allows configuration of parameters for the simulation, including the number of iterations and dataset dimensions.

- <ins>Using the script</ins>:

1. Clone the repository,
2. Install the latest version of R available, download RStudio or some development environment for R,
3. Run the toy example set in the script,
4. Modify the set parameters (optional)

# Load script

`source("Simulate.R")`

# Example for parameter definition:

```
x <- 10
Sigma <- matrix(0, x, x)
diag(Sigma) <- 1
nnull <- round(.3*x, 0)
beta1 <- c(rep(0, x-nnull), runif(nnull, -0.5, 0.5))
beta2 <- c(runif(nnull, -0.5, 0.5), rep(0, x-nnull))
```

# Run simulation

```
results <- Simulate(
  sigma = Sigma,
  beta.list = list(beta1, beta2),
  p = x,
  n.train = 150,
  n.test = 2000,
  R = 10
)
```

# View results

`print(results)`

# Script outputs:

- <ins>Misclassification rates</ins>: mean and standard deviation for all models. 
- <ins>Precision and Recall</ins>: For LASSO and Elastic Net models.
- <ins>Metadata</ins>: Total time, number of attempts and successful iterations.

# If you need to install a dependency, you can do so as follows:

### <ins>Example</ins>: install the dependencies 'glmnet' and 'randomForest'

`install.packages(c("glmnet", "randomForest"))`

# Author

Molina, Agustin

# Year

2024
