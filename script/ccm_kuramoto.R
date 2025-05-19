library(ggplot2)
library(dplyr)
library(rEDM)
library("data.table")


data <- fread("path_to_oscillatory_data.txt")

L = nrow(data)
N = ncol(data)

L


data$time <- ((1:L) -1)* 0.63

data$time

data <- data %>% mutate(x1 = V1, x2 = V2) %>% select(time, x1, x2) 

libsize=L-1
ccm <- CCM(dataFrame = data, 
    E = 2,
    Tp = 0,
    columns = "x1",
    target = "x2", 
    libSizes = libsize, 
    sample = 1, 
    showPlot = FALSE)

ccm ##x1:x2 corresponds to the coupling from 2 to 1

