cat("=======================================================",
    "\nIris Model Applied to a Dataset to Predict Iris Species",
    "\n=======================================================\n\n")

library(rpart)
library(magrittr)
library(dplyr)
library(rattle)

load("iris_model.RData")

read.csv("data.csv") %T>%
  assign('ds', ., envir=.GlobalEnv) %>%
  predict(m$finalModel, newdata=., type="class") %>%
  as.data.frame() %>%
  cbind(Actual=ds$Species) %>%
  set_names(c("Predicted", "Actual")) %>%
  select(Actual, Predicted) %>%
  mutate(Error=ifelse(Predicted==Actual, "", "<----")) %T>%
  print() ->
ev
  
# Produce confusion matrix using Rattle?

cat("\n================",
    "\nConfusion Matrix",
    "\n================\n\n")

per <- errorMatrix(ev$Actual, ev$Predicted)

per/100

# Calculate the overall error percentage.

cat(sprintf("\nOverall error: %d%%\n", 100-sum(diag(per), na.rm=TRUE)))

# Calculate the averaged class error percentage.

cat(sprintf("Average class error: %.1f%%\n", mean(per[,"Error"], na.rm=TRUE)))
