library("TSA")  
library("lmtest")

#Q1
n = 200
set.seed(1)
y_1 = y_2 = a = rnorm(n)
params = c(0.01,0.03,-0.03)
for(t in 3:n) y_1[t] = params[1] + params[2] * y_1[t-1] + params[3] * y_1[t-2] + a[t]
plot.ts(y_1)

#Q2
x_1 = c(c(0),y_1[1:length(y_1)-1])
x_2 = c(c(0,0),y_1[1:198])
data_1 = data.frame(y_1, x_1, x_2,a)
lmfit = lm(formula = y_1~x_1+x_2+a,data=data_1)
summary(lmfit)
resettest(lmfit,power = 2:3,type='regressor',data=data_1)

#Q3
params = c(0.01,0.03,-0.03,0.01)
for(t in 3:n) y_2[t] = params[1] + params[2] * y_2[t-1] + params[3] * y_2[t-2] + params[4] * y_2[t-1]^2 + a[t]
plot.ts(y_2)

#Q4
x_1 = c(c(0),y_2[1:length(y_2)-1])
x_2 = c(c(0,0),y_2[1:198])
x_12 = c(c(0),y_2[1:length(y_2)-1]^2)
data_2 = data.frame(y_2, x_1,x_2,x_12,a)
lmfit = lm(formula = y_2~x_1+x_2+x_12+a,data=data_2)
summary(lmfit)
resettest(lmfit,power = 2:3,type='regressor',data=data_2)

