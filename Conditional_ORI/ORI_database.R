wd = "~/Documents/GitHub/FHAI/Conditional_ORI"
setwd(wd)

fn <- "data-ori.csv"
fp <- file.path(wd, fn)

data <- read.csv(fp)

# Print the contents of the CSV file
head(data)
