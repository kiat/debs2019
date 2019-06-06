library(extrafont);
loadfonts();

pdf("/home/saeed/sectors_density.pdf",paper="special", height=3.2, width=3.5,family="Helvetica");

par(oma=c(0,0,0,0),mar=c(3.9,4,0.5,0.5))

x <- seq(from = 0, to = 10, length.out = 100)

y <- dnorm(x, mean = 2, sd =1.2) + dnorm(x, mean = 5 , sd = 0.7) + dnorm(x, mean = 8 , sd = 1)

# Plot it

plot(x, y, type="l", ylab = "Density of Points", xlab = "Radius of Points to Center")

dev.off()