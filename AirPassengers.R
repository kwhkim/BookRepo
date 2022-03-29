library(readr)

data(AirPassengers)

dat = data.frame(y=as.matrix(AirPassengers), date=zoo::as.Date(time(AirPassengers)))
dat

write_csv(file='./BookRepo/data/AirPassengers.csv', dat)
