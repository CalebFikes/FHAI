wd = "~/Documents/GitHub/FHAI/Conditional_ORI"
setwd(wd)

fn <- "data-ori.csv"
fp <- file.path(wd, fn)

data <- read.csv(fp)
data$SEX <- as.factor(data$SEX)
data$SOURCE <- as.factor(data$SOURCE)
dm = dim(data)
data = data.frame(data)

# Print the contents of the CSV file
head(data)
#View(data)
dim(data)
typeof(data)

train = sample(1:dm[1], .9 * dm[1])


#LOGISTIC REGRESSION MODEL
model <- glm(SOURCE ~ .,
						 data = data,
						 subset = train,
						 family = binomial)

# Summary of the model
summary(model)

#test
pred <- predict(model, newx = data[setdiff(1:dm[1], train),1:dm[2]])
pred_lab = factor(ifelse(pred > 0.5, "in", "out"))
accuracy <- sum(pred_lab == data[setdiff(1:dm[1], train),dm[2]]) / length(pred)
