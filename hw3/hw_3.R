library("TSA")  
n = 500
set.seed(1)
x = a = rnorm(n)
params = c(0.3,0.7,-0.8,0.1)
for(t in 3:n) x[t] = params[1] + params[2] * x[t-1] + a[t] + params[3] * a[t-1] + params[4] * a[t-2]

p = ''
for (i in 1:length(x)) p = paste(p,',',as.character(x[i])) 

p
result = eacf(x, 10, 10)
