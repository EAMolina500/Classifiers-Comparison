#=========================================
# File: classification_methods_comparison_tool.R
# 
# Description: 
# This script provides a tool for comparing classification methods, 
# and all the functions involved.
#
# Dependencies:
# - Libraries: Matrix, MASS, randomForest, plyr, glmnet, mvnfast, base, nnet
#
# Author: Molina, Agustin
# Date: December 2024
#=========================================

rm(list = ls())

library(Matrix)
library(MASS)
library(randomForest)
library(plyr) 
library(glmnet)
library(mvnfast)
library(base)
library(nnet)

##################################
## Auxiliar functions
##################################

#' GenerateDataX
#' 
#' Generates a matrix of multivariate normal random variables.
#' 
#' @param n Integer. Number of samples.
#' @param p Integer. Number of features.
#' @param sigma Matrix. Covariance matrix for the multivariate normal distribution.
#' @return Matrix of dimensions (n, p) containing generated data, or NULL on error.
GenerateDataX <- function(n, p, sigma) {
  tryCatch({
    mvrnorm(n, rep(1, p), sigma)
  }, error = function(e) {
    message("Error in 'GenerateDataX':", e$message)
    NULL
  })
}

#' GenerateDataY
#' 
#' Generates class labels based on logits computed using input data and coefficients.
#' 
#' @param x Matrix. Input data (samples x features).
#' @param beta.list List. Coefficients for each class.
#' @return Vector of class labels for each sample, or NULL on error.
GenerateDataY <- function(x, beta.list) {
  tryCatch({
    logits <- sapply(1:length(beta.list), function(i) x %*% beta.list[[i]])
    exp_logits <- exp(logits)
    den <- rowSums(exp_logits) + 1
    P <- cbind(exp_logits / den, 1 / den)
    max.col(P)
  }, error = function(e) {
    message("Error in 'GenerateDataY':", e$message)
    NULL
  })
}

#' Evaluate
#' 
#' Calculates the classification error rate.
#' 
#' @param y.true Vector. True class labels.
#' @param y.pred Vector. Predicted class labels.
#' @return Numeric value representing the error rate, or NA on error.
Evaluate <- function(y.true, y.pred) {
  tryCatch({
    mean(y.true != y.pred)
  }, error = function(e) {
    message("Error in 'Evaluate':", e$message)
    NA
  })
}

#' PrepareData
#' 
#' Formats data matrix and assigns column names.
#' 
#' @param x Matrix. Input data.
#' @param y Vector. Class labels (not used in this function).
#' @param n Integer. Number of samples.
#' @param p Integer. Number of features.
#' @return Matrix with named columns, or NULL on error.
PrepareData <- function(x, y, n, p) {
  tryCatch({
    x.names <- paste0("X", seq_len(p))
    X <- matrix(x, n, dimnames = list(NULL, x.names))
    return(X)
  }, error = function(e) {
    message("Error in 'PrepareData':", e$message)
    NULL
  })
}

#' GetLambdaMinByCrossValidation
#' 
#' Finds the optimal lambda using cross-validation for LASSO or Elastic Net.
#' 
#' @param method String. "LML" for LASSO, "LME" for Elastic Net.
#' @param x.train Matrix. Training data.
#' @param y.train Vector. Training labels.
#' @return Numeric value for the optimal lambda, or NULL on error.
GetLambdaMinByCrossValidation <- function(method, x.train, y.train) {
  tryCatch({
    alpha <- switch(
      method,
      "LML" = 1,
      "LME" = 0.5,
      stop("Unsupported method. Use LML (lasso) or LME (elastic net).")
    )
    cv.fit <- cv.glmnet(x = x.train, y = as.factor(y.train), alpha = alpha, family = "multinomial", nfolds = 3)
    lambda.min <- cv.fit$lambda.min
    return(lambda.min)
  }, error = function(e) {
    message("Error in 'GetLambdaMinByCrossValidation':", e$message)
    NULL
  })
}

##################################
## Functions to Train a model and Predict using a model
##################################

#' Train
#' 
#' Trains a model using the specified method.
#' 
#' @param method String. One of "LDA", "LG", "LML", "RF", or "LME".
#' @param x.train Matrix. Training data.
#' @param y.train Vector. Training labels.
#' @return Trained model object, or NULL on error.
Train <- function(method, x.train, y.train) {
  tryCatch({
    switch(
      method,
      "LDA" = lda(x.train, y.train, method="mle"),
      "LG" = multinom(formula = y.train ~ ., data = as.data.frame(x.train), MaxNWts = 1500),
      "LML" = glmnet(x.train, y.train, alpha = 1, family = "multinomial"),
      "RF" = randomForest(x.train, as.factor(y.train), importance = TRUE),
      "LME" = glmnet(x.train, y.train, alpha = .5, family = "multinomial"),
      stop(paste("Error in 'Train': Unsupported method '", method, "'"))
    )
  }, error = function(e) {
    message(paste("Error in 'Train' with method '", method, "' :", e$message))
    NULL
  })
}

#' Predict
#' 
#' Makes predictions using a trained model.
#' 
#' @param method String. Method used for training.
#' @param model Object. Trained model.
#' @param x.test Matrix. Test data.
#' @param lambda.min Numeric. Regularization parameter for LASSO/Elastic Net.
#' @return Vector of predicted labels, or NULL on error.
Predict <- function(method, model, x.test, lambda.min = NULL) {
  tryCatch({
    switch(
      method,
      "LDA" = predict(model, x.test)$class,
      "LG" = predict(model, newdata = as.data.frame(x.test), type = "class"),
      "LML" = predict(model, x.test, s = lambda.min, type = "class"),
      "RF" = predict(model, x.test, type = "class"),
      "LME" = predict(model, x.test, s = lambda.min, type = "class"),
      stop(paste("Error in 'Predict': Unsupported method: '", method, "'"))
    )
  }, error = function(e) {
    message(paste("Error in 'Predict' with method '", method, "':", e$message))
    NULL
  })
}

##################################
## Functions to calculate precision and recall measures
##################################

#' ExtractCoefficients
#' 
#' Extracts the coefficients from a trained model for a specific lambda value.
#' 
#' @param model Object. The trained model containing coefficients.
#' @param lambda.min Numeric. The lambda value for which coefficients are extracted.
#' @return Numeric vector. Unlisted coefficients across all classes.
ExtractCoefficients <- function(model, lambda.min) {
  beta.list <- lapply(1:length(model$beta), function(k) {
    unname(model$beta[[k]][, which(model$lambda == lambda.min)])
  })
  
  unlist(beta.list)
}

#' CalculatePrecisionRecall
#' 
#' Computes precision and recall metrics based on true and estimated coefficients.
#' 
#' @param beta.true.list List. True beta coefficients for each class.
#' @param beta.estim.list List. Estimated beta coefficients for each class.
#' @return List. Contains `recall` and `precision` values.
CalculatePrecisionRecall <- function(beta.true.list, beta.estim.list) {
  beta.true <- unlist(beta.true.list)
  beta.estim <- unlist(beta.estim.list)
  
  matches <- sum(beta.true != 0 & beta.estim != 0)
  real.nonzeros <- sum(beta.true != 0)
  estim.nonzeros <- sum(beta.estim != 0)
  
  recall <- matches / real.nonzeros
  precision <- matches / estim.nonzeros
  
  list(
    recall = recall,
    precision = precision
  )
}

##################################
## Functions to get misclasification rate 
##################################

#' MiscRateByLDA
#' 
#' Calculates the misclassification rate using Linear Discriminant Analysis (LDA).
#' 
#' @param x.train Matrix. Training feature data.
#' @param y.train Factor. Training labels.
#' @param x.test Matrix. Test feature data.
#' @param y.test Factor. Test labels.
#' @return Numeric. Misclassification rate for LDA.
MiscRateByLDA <- function(x.train, y.train, x.test, y.test) {
  lda.model <- Train("LDA", x.train, y.train)
  lda.prediction <- Predict("LDA", lda.model, x.test)
  Evaluate(y.test, lda.prediction)
}

#' MiscRateByLG
#' 
#' Calculates the misclassification rate using Logistic Regression (LG).
#' 
#' @param x.train Matrix. Training feature data.
#' @param y.train Factor. Training labels.
#' @param x.test Matrix. Test feature data.
#' @param y.test Factor. Test labels.
#' @param n Integer. Number of observations in the training data.
#' @param m Integer. Number of observations in the test data.
#' @param p Integer. Number of features in the dataset.
#' @return Numeric. Misclassification rate for Logistic Regression.
MiscRateByLG <- function(x.train, y.train, x.test, y.test, n, m, p) {
  X.train <- PrepareData(x.train, y.train, n, p)
  lg.model <- Train("LG", X.train, y.train)
  
  X.test <- PrepareData(x.test, y.test, m, p)
  lg.prediction <- Predict("LG", lg.model, X.test)
  
  Evaluate(y.test, lg.prediction)
}

#' MiscRateByLML
#' 
#' Calculates the misclassification rate using LASSO logistic regression.
#' Extracts beta coefficients for further analysis.
#' 
#' @param x.train Matrix. Training feature data.
#' @param y.train Factor. Training labels.
#' @param x.test Matrix. Test feature data.
#' @param y.test Factor. Test labels.
#' @return List. Contains `misc.rate` (misclassification rate) and `beta.hat` (estimated coefficients).
MiscRateByLML <- function(x.train, y.train, x.test, y.test) {
  lml.model <- Train("LML", x.train, y.train)
  lml.lmin <- GetLambdaMinByCrossValidation("LML", x.train, y.train)
  lml.prediction <- Predict("LML", lml.model, x.test, lml.lmin)
  
  misc.rate <- Evaluate(y.test, lml.prediction)
  beta.hat <- ExtractCoefficients(lml.model, lml.lmin)
  
  list(
    misc.rate = misc.rate,
    beta.hat = beta.hat
  )
}

#' MiscRateByRF
#' 
#' Calculates the misclassification rate using Random Forest (RF).
#' 
#' @param x.train Matrix. Training feature data.
#' @param y.train Factor. Training labels.
#' @param x.test Matrix. Test feature data.
#' @param y.test Factor. Test labels.
#' @return Numeric. Misclassification rate for Random Forest.
MiscRateByRF <- function(x.train, y.train, x.test, y.test) {
  rf.model <- Train("RF", x.train, y.train)
  rf.prediction <- Predict("RF", rf.model, x.test)
  Evaluate(y.test, rf.prediction)
}

#' MiscRateByLME
#' 
#' Calculates the misclassification rate using Elastic Net logistic regression.
#' Extracts beta coefficients for further analysis.
#' 
#' @param x.train Matrix. Training feature data.
#' @param y.train Factor. Training labels.
#' @param x.test Matrix. Test feature data.
#' @param y.test Factor. Test labels.
#' @return List. Contains `misc.rate` (misclassification rate) and `beta.hat` (estimated coefficients).
MiscRateByLME <- function(x.train, y.train, x.test, y.test) {
  lme.model <- Train("LME", x.train, y.train)
  lme.lmin <- GetLambdaMinByCrossValidation("LME", x.train, y.train)
  lme.prediction <- Predict("LME", lme.model, x.test, lme.lmin)
  
  misc.rate <- Evaluate(y.test, lme.prediction)
  beta.hat <- ExtractCoefficients(lme.model, lme.lmin)
  
  list(
    misc.rate = misc.rate,
    beta.hat = beta.hat
  )
}

##################################
## Main function to execute the simulation 
##################################

#' Simulate
#' 
#' Executes a simulation to evaluate classification models and calculate 
#' precision-recall metrics for LASSO and Elastic Net logistic regression.
#' 
#' @param sigma Matrix. Covariance matrix used to generate the predictors (x).
#' @param beta.list List. True coefficients for each class to generate the responses (y).
#' @param p Integer. Number of predictors. Default is 10.
#' @param n.train Integer. Number of training samples. Default is 150.
#' @param n.test Integer. Number of test samples. Default is 2000.
#' @param R Integer. Number of simulation repetitions to perform. Default is 50.
#' 
#' @return A list containing the following elements:
#' \describe{
#'   \item{total.time}{Duration of the simulation.}
#'   \item{misc.rate.mean}{Mean misclassification rates for all methods across simulations.}
#'   \item{standard.deviation}{Standard deviation of misclassification rates for all methods.}
#'   \item{misc.rate}{Matrix of misclassification rates for each method in each simulation.}
#'   \item{attempts}{Number of total attempts made to complete the simulations.}
#'   \item{completed}{Number of successfully completed simulations.}
#'   \item{lasso.recall}{Mean recall for LASSO across simulations.}
#'   \item{lasso.precision}{Mean precision for LASSO across simulations.}
#'   \item{elastic.net.recall}{Mean recall for Elastic Net across simulations.}
#'   \item{elastic.net.precision}{Mean precision for Elastic Net across simulations.}
#' }
#' 
#' @details 
#' The function performs multiple simulations to evaluate the performance of different 
#' classification models: Linear Discriminant Analysis (LDA), Logistic Regression (LG), 
#' LASSO Logistic Regression (LML), Random Forest (RF), and Elastic Net Logistic Regression (LME).
#' 
#' For LASSO and Elastic Net, it calculates precision and recall based on the estimated 
#' coefficients compared to the true coefficients. Simulations terminate either when 
#' the specified number of repetitions (\code{R}) is reached or the maximum number of 
#' attempts (\code{R * 10}) is exceeded.
#' 
#' @examples
#' sigma <- diag(10) # Example covariance matrix
#' beta.list <- list(rep(1, 10), rep(0, 10)) # Example coefficients
#' results <- Simulate(sigma, beta.list, p = 10, n.train = 100, n.test = 500, R = 10)
#' 
#' @export
Simulate <- function(
    sigma, beta.list, p = 10, n.train = 150, n.test = 2000, R = 50
) {

  misc.rate <- array(NA, dim = c(R, 5))
  completed <- 0
  attempts <- 0
  max.attempts <- R * 10
  lasso.metrics <- matrix(NA, nrow = R, ncol = 2)
  elastic.net.metrics <- matrix(NA, nrow = R, ncol = 2)
  
  t0 <- Sys.time()
  
  while (completed < R && attempts < max.attempts) {
    attempts <- attempts + 1
    tryCatch({
      message("Iteration: ", attempts)
      message("Completed: ", completed)
      
      x.train <- GenerateDataX(n.train, p, sigma)
      x.test <- GenerateDataX(n.test, p, sigma)
      
      y.train <- GenerateDataY(x.train, beta.list)
      y.test <- GenerateDataY(x.test, beta.list)
      
      misc.rate[completed + 1, 1] <- MiscRateByLDA(x.train, y.train, x.test, y.test)
      misc.rate[completed + 1, 2] <- MiscRateByLG(x.train, y.train, x.test, y.test, n.train, n.test, p)
      
      lasso.results <- MiscRateByLML(x.train, y.train, x.test, y.test)
      misc.rate[completed + 1, 3] <- lasso.results$misc.rate
      
      lasso.pr <- CalculatePrecisionRecall(beta.list, lasso.results$beta.hat)
      lasso.metrics[completed + 1, 1] <- lasso.pr$recall
      lasso.metrics[completed + 1, 2] <- lasso.pr$precision
      
      misc.rate[completed + 1, 4] <- MiscRateByRF(x.train, y.train, x.test, y.test)
      
      elastic.net.results <- MiscRateByLME(x.train, y.train, x.test, y.test) 
      misc.rate[completed + 1, 5] <- elastic.net.results$misc.rate
      
      elastic.net.pr <- CalculatePrecisionRecall(beta.list, elastic.net.results$beta.hat)
      elastic.net.metrics[completed + 1, 1] <- elastic.net.pr$recall
      elastic.net.metrics[completed + 1, 2] <- elastic.net.pr$precision
      
      completed <- completed + 1
      
    }, error = function(e) {
      message("Error in attempt: ", attempts, " - ", e$message)
    })
  }
  
  lasso.recall <- mean(lasso.metrics[, 1], na.rm = TRUE)
  lasso.precision <- mean(lasso.metrics[, 2], na.rm = TRUE)
  elastic.net.recall <- mean(elastic.net.metrics[, 1], na.rm = TRUE)
  elastic.net.precision <- mean(elastic.net.metrics[, 2], na.rm = TRUE)
  
  list(
    total.time = Sys.time() - t0,
    misc.rate.mean = round(colMeans(misc.rate, na.rm = TRUE), 3),
    standard.deviation = round(apply(misc.rate, 2, sd, na.rm = TRUE), 3),
    misc.rate = misc.rate,
    attempts = attempts,
    completed = completed,
    lasso.recall = round(lasso.recall, 3),
    lasso.precision = round(lasso.precision, 3),
    elastic.net.recall = round(elastic.net.recall, 3),
    elastic.net.precision = round(elastic.net.precision, 3)
  )
}



######################################
## Initialize values
######################################

x <- 10
Sigma <- matrix(0, x, x)
diag(Sigma) <- 1
nnull <- round(.3*x, 0)

set.seed(2)

beta1 <- c(rep(0, x-nnull), runif(nnull, -0.5, 0.5))
beta2 <- c(runif(nnull, -0.5, 0.5), rep(0, x-nnull))

results <- Simulate(
  sigma = Sigma,
  beta.list = list(beta1, beta2),
  p = x,
  n.train = 150,
  n.test = 2000,
  R = 10
)

message("Completed: ", results$completed)

######################################
## Printing results
######################################

cat("3 clases - R 10 - n 150 - m 2000 - p 10")
cat("\nTiempo total\n")
print(results$total.time)
cat("\nPromedio de mal clasificacion por metodo\n")
print(results$misc.rate.mean)
cat("\nSD por metodo\n")
print(results$standard.deviation)
cat("\nLasso recall\n")
print(results$lasso.recall)
cat("\nLasso precision\n")
print(results$lasso.precision)
cat("\nElastic net recall\n")
print(results$elastic.net.recall)
cat("\nElastic net precision\n")
print(results$elastic.net.precision)

data <- stack(data.frame(results$misc.rate))
label <- c("LDA", "LG", "LML", "RF", "LME")

boxplot(data$values ~ data$ind, names=label, 
        xlab='Boxplots de 5 métodos para p=10', 
        ylab='Tasas de mal clasificación para R=10 réplicas', 
        main='Desempeño de clasificadores lineales para K=3 clases')


