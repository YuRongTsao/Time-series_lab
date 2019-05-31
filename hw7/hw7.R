library("mvtnorm")
library("MTS")
library("psych")
#Q1
n=200
f = rmvnorm(n=n, mean=mean, sigma=var)
a = rmvnorm(n=n, mean=mean, sigma=var) 
mean = matrix(c(0,0),nrow = 2,ncol = 1,byrow = TRUE)
var = matrix(c(2,1,1,1),nrow = 2,ncol = 2,byrow = TRUE)
fi0 = matrix(c(0.2,0.4),nrow = 1,ncol = 2,byrow = TRUE)
fi = matrix(c(0.2,0.3,-0.6,1.1),nrow = 2,ncol = 2,byrow = TRUE)

for(t in 2:n) f[t,] = fi0 + t(fi %*% f[t-1,]) + a[t,]

plot(f)

#Q2
model = VARMA(f,p=1,q=1,include.mean=FALSE)

#Q3
mean = matrix(c(0,0,0),nrow = 3,ncol = 1,byrow = TRUE)
var = matrix(c(1,0,0,0,1,0,0,0,1),nrow = 3,ncol = 3,byrow = TRUE)
b = rmvnorm(n=n, mean=mean, sigma=var)
x = rmvnorm(n=n, mean=mean, sigma=var)
para = matrix(c(0.9,0,1.2,0),nrow = 2,ncol = 2,byrow = TRUE)

for(t in 1:n) x[t,] =  cbind(t(para %*% f[t,]),c(0)) + b[t,]

plot(x)

#Q4
pca = prcomp(x, center = TRUE, scale = TRUE)
pc = pca$rotation[,-3]
pc
factor_model <- factor.model(pc)
round(factor_model,2)
summary(factor_model)
