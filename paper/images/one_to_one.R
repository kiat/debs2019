library(extrafont);
loadfonts();
path<-getwd();

# Train and validation accuracy
train_acc <-c(4.0,30.1,42.3,54.6,61.8,67.6,72.5,75.9,78.7,81.7,84.7,87.2)
val_acc <-c(3.9,32.3,47.2,62.9,69.2,74.9,77.9,81.2,82.5,85.4,88.3,92.7)

# Train and validation loss
train_loss <-c(3.347,2.224,1.742,1.408,1.1798,0.98703,0.8687,0.7552,0.6347,0.5813,0.5345,0.4935)
val_loss <- c(3.339,2.1006,1.5959,1.2513,1.0288,0.88,0.7654,0.6983,0.6212,0.5836,0.5516,0.4953)

range_value=seq(1,12)

pdf("one_to_one.pdf", height=2.2, width=3.5,family="Helvetica");

plot_colors <- c("blue","red");

par(mfrow=c(1,2),oma=c(0,0,0,0),mar=c(2.5,2,0.7,0.5))

#Train and validation accuracy
plot(train_acc, type="l", col = plot_colors[1], lwd=1, xlab = "", ylab = "",ylim=c(0,100), main = "Object-net",cex.main=0.5,xaxs="i", yaxs="i",axes=FALSE, frame.plot=TRUE)
lines( val_acc, col = plot_colors[2], lwd=1)

axis(side=1, at=seq(1,12),labels=range_value , lwd.ticks=0.3,cex.axis=0.35,mgp=c(3, 0.2, 0))
axis(side=2,  lwd.ticks=0.3,cex.axis=0.35,mgp=c(3, 0.6, 0),las=1)
title(ylab="Accuracy", line=1, cex.lab=0.5, family="Helvetica")
title(xlab="Epochs", line=1, cex.lab=0.5, family="Helvetica")
legend(4.5,18,c("Training Accuracy","Validation Accuracy"), lwd=c(1,1), col=plot_colors, y.intersp=1,pch=16,cex = 0.4)

# Train and validation loss
plot(train_loss, type="l", col = plot_colors[1], lwd=1, xlab = "", ylab = "",ylim=c(0,4), main = "Object-net",cex.main=0.5,xaxs="i", yaxs="i",axes=FALSE, frame.plot=TRUE)
lines( val_loss, col = plot_colors[2], lwd=1)

axis(side=1, at=seq(1,12),labels=range_value , lwd.ticks=0.3,cex.axis=0.35,mgp=c(3, 0.2, 0))
axis(side=2,  lwd.ticks=0.3,cex.axis=0.35,mgp=c(3, 0.6, 0),las=1)
title(ylab="Loss", line=1, cex.lab=0.5, family="Helvetica")
title(xlab="Epochs", line=1, cex.lab=0.5, family="Helvetica")
legend(5.5,3.9,c("Training Loss","Validation Loss"), lwd=c(1,1), col=plot_colors, y.intersp=1,pch=16,cex = 0.4)



dev.off()
