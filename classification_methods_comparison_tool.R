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

GenerateDataX <- function(n, p, sigma) {
  tryCatch({
    mvrnorm(n, rep(1, p), sigma)
  }, error = function(e) {
    message("Error in 'GenerateDataX':", e$message)
    NULL
  })
}

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

Evaluate <- function(y.true, y.pred) {
  tryCatch({
    mean(y.true != y.pred)
  }, error = function(e) {
    message("Error in 'Evaluate':", e$message)
    NA
  })
}

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

ExtractCoefficients <- function(model, lambda.min) {
  beta.list <- lapply(1:length(model$beta), function(k) {
    unname(model$beta[[k]][, which(model$lambda == lambda.min)])
  })
  
  unlist(beta.list)
}

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

MiscRateByLDA <- function(x.train, y.train, x.test, y.test) {
  lda.model <- Train("LDA", x.train, y.train)
  lda.prediction <- Predict("LDA", lda.model, x.test)
  Evaluate(y.test, lda.prediction)
}

MiscRateByLG <- function(x.train, y.train, x.test, y.test, n, m, p) {
  X.train <- PrepareData(x.train, y.train, n, p)
  lg.model <- Train("LG", X.train, y.train)
  
  X.test <- PrepareData(x.test, y.test, m, p)
  lg.prediction <- Predict("LG", lg.model, X.test)
  
  Evaluate(y.test, lg.prediction)
}


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


MiscRateByRF <- function(x.train, y.train, x.test, y.test) {
  rf.model <- Train("RF", x.train, y.train)
  rf.prediction <- Predict("RF", rf.model, x.test)
  Evaluate(y.test, rf.prediction)
}


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


