# High-Dimensional Model Simulation Tool Overview

### Undergraduate thesis project at the National University of Río Cuarto.

This repository contains an R script that implements a simulation tool designed to evaluate the performance of classification models in high-dimensional settings. The tool is developed as part of the undergraduate thesis titled *"Multinomial Logistic Regression in High Dimensions."*

---

## Repository Structure

- **`scripts/`**: Contains the main R script used for the simulation tool.  
- **`thesis/`**: Includes the thesis files in `.pdf` and `.tex` formats, as well as associated images and additional resources used in the document.

---

## Purpose

The core problem addressed by this project is the evaluation of classification algorithms when the number of predictors (features) is large compared to the sample size. High-dimensional datasets pose challenges to traditional methods, making it critical to assess their accuracy, robustness, and parameter recovery capabilities.

---

## Features

### The script provides:

- **Synthetic Data Generation**: Produces training and testing datasets with user-defined properties, including the number of features, sample size, and covariance structure.  
- **Model Evaluation**: Implements multiple classification models:  
  - Linear Discriminant Analysis (LDA)  
  - Multinomial Logistic Regression (MLR)  
  - Penalized Regression models (LASSO and Elastic Net)  
  - Random Forest  
- **Metrics Calculation**: Quantifies model performance using misclassification rates and parameter recovery metrics, such as precision and recall for penalized regression models.  
- **Simulation Control**: Allows configuration of parameters for the simulation, including the number of iterations and dataset dimensions.  

---

## Usage Instructions

1. Clone the repository:  
```
git clone https://github.com/your-repo-name.git
```
2. Install the latest version of R and an IDE like RStudio.
3. Load the main script using
```
source("scripts/classification_methods_comparison_tool.R")
```
4. Run the toy example included in the script or modify the parameters as needed.

---

## Example: Parameter Definition:

```
x <- 10
Sigma <- matrix(0, x, x)
diag(Sigma) <- 1
nnull <- round(.3*x, 0)
beta1 <- c(rep(0, x-nnull), runif(nnull, -0.5, 0.5))
beta2 <- c(runif(nnull, -0.5, 0.5), rep(0, x-nnull))
```

---

## Run simulation

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

---

## View results

```
print(results)
```

## Outputs included:

- <ins>Misclassification rates</ins>: mean and standard deviation for all models. 
- <ins>Precision and Recall</ins>: For LASSO and Elastic Net models.
- <ins>Metadata</ins>: Total time, number of attempts and successful iterations.

---

## Dependencies:

To install dependencies, use the following command:

```
install.packages(c("glmnet", "randomForest"))
```

---

## Author

Molina, Agustin

## Year

2024
