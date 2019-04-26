x <- seq(from = 0, to = 10, length.out = 100)

y <- dnorm(x, mean = 2, sd =1.2) + dnorm(x, mean = 5 , sd = 0.7) + dnorm(x, mean = 8 , sd = 1)

# Plot it
plot(x, y, type="l", ylab = "Density of Points", xlab = "Radius of points to Center")

