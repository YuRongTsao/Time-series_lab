library(fExpressCertificates)
#Q1
t <- 0:100  # time
sig2 <- 0.01

nsim <- 10000
X <- matrix(rnorm(n = nsim * (length(t) - 1), sd = sqrt(sig2)), nsim, length(t) - 
              1)
X <- cbind(rep(0, nsim), t(apply(X, 1, cumsum)))
plot(t, X[1, ], xlab = "time", ylab = "phenotype", ylim = c(-2, 2), type = "l")
apply(X[2:nsim, ], 1, function(x, t) lines(t, x), t = t)


#Q2
m_t <- apply(X, 1, min, na.rm=TRUE)
cdf <- ecdf(m_t)
plot(cdf)
x = -3
y <- cdf(x) 
print(y)

#Q3
t <- 0:10000  # time
sig2 <- 0.01
nsim <- 10000
X <- matrix(rnorm(n = nsim * (length(t) - 1), sd = sqrt(sig2)), nsim, length(t) -               1)
X <- cbind(rep(0, nsim), t(apply(X, 1, cumsum)))

m_t <- apply(X, 1, min, na.rm=TRUE)
cdf <- ecdf(m_t)
plot(cdf)
x = -3
y <- cdf(x) 
print(y)
