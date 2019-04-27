library(extrafont);
loadfonts();
path<-getwd();

model_names <-c('a= 2-Layer CNN on projected data to 2D (Single View) and Object Segmentation with 3D DBSCAN', 'b= 2-Layer CNN on projected data to 2D (Using perspective projection) and Object Segmentation with 3D DBSCAN', 'c= 4-Layer CNN on projected data to 2D (Single View) and Object Segmentation with 3D DBSCAN', 'd= 4-Layer CNN on projected data to 2D (Using perspective projection) and Object Segmentation with 3D DBSCAN')
model <-c('a','b','c','d')
accuracy <-c(13.35, 39.79, 45.12, 47.06)
time<-c(3.958, 1.118, 1.246, 0.924)
precision<-c(17.43, 50.78, 57.76,61.06)
recall<-c(17.92, 60.09, 58.9,63.08)
scene_evaluated<-c(146, 488, 446, 500)


plot_colors <- c("blue","green","red","grey90");

pdf("evaluation2.pdf", height=2.2, width=8,family="Helvetica");

model_range<-c(1,2,3,4)

plot_colors <- c("blue","green","red","darkblue","grey90");

par(mfrow=c(1,5),oma=c(0,0,0,0),mar=c(6.2,3.5,0.5,0.5))

#Accuracy
barplot(accuracy,type = "l", col = plot_colors[1], xlab = "", ylab = "", main = "",lwd=2,pch=15,ylim=c(0,50),xaxs="i", yaxs="i",axes = FALSE);

#grid(nx = 20);

axis(side=1, at=seq(1,4),labels=model , lwd.ticks=1,cex.axis=0.7)
axis(side=2,  lwd.ticks=1,cex.axis=0.7)

#text(seq(1,4), par("usr")[3]-0.25, 
#     srt = 60, adj= 1, xpd = TRUE,
#     labels = model, cex=0.65)

title(ylab="Accuracy(Percent)", line=2, cex.lab=0.8, family="Helvetica")
legend(x=0,y=-0.5,legend=model_names[0],col='black',pch=16,bty="n",xpd=NA)
abline(h=0,col="black",lwd=2)


#precision
barplot(precision,type = "l", col = plot_colors[2], xlab = "", ylab = "", main = "",lwd=2,pch=15,ylim=c(0,70),xaxs="i", yaxs="i",axes = FALSE);

#grid(nx = 20);

axis(side=1, at=seq(1,4),labels=model , lwd.ticks=1,cex.axis=0.7)
axis(side=2,  lwd.ticks=1,cex.axis=0.7,las=1)
#text(seq(1,4), par("usr")[3]-0.25, 
#     srt = 60, adj= 1, xpd = TRUE,
#     labels = model, cex=0.65)
title(ylab="Precision(Percent)", line=2, cex.lab=0.8, family="Helvetica")
abline(h=0,col="black",lwd=2)

#recall
barplot(recall,type = "l", col = plot_colors[3], xlab = "", ylab = "", main = "",lwd=2,pch=15,ylim=c(0,70),xaxs="i", yaxs="i",axes = FALSE);

#grid(nx = 20);

axis(side=1, at=seq(1,4),labels=model , lwd.ticks=1,cex.axis=0.7)
axis(side=2,  lwd.ticks=1,cex.axis=0.7,las=1)
#text(seq(1,4), par("usr")[3]-0.25, 
#     srt = 60, adj= 1, xpd = TRUE,
#     labels = model, cex=0.65)

title(ylab="Recall(Percent)", line=2, cex.lab=0.8, family="Helvetica")
abline(h=0,col="black",lwd=2)


#scene_evaluated
barplot(scene_evaluated,type = "l", col = plot_colors[4], xlab = "", ylab = "", main = "",lwd=2,pch=15,ylim=c(0,500),xaxs="i", yaxs="i",axes = FALSE);

#grid(nx = 20);

axis(side=1, at=seq(1,4),labels=model , lwd.ticks=1,cex.axis=0.7)
axis(side=2,  lwd.ticks=1,cex.axis=0.7,las=1)
#text(seq(1,4), par("usr")[3]-0.25, 
#     srt = 60, adj= 1, xpd = TRUE,
#     labels = model, cex=0.65)

title(ylab="Scene Evaluated", line=2, cex.lab=0.8, family="Helvetica")
abline(h=0,col="black",lwd=2)


#time
barplot(time,type = "l", col = plot_colors[5], xlab = "", ylab = "", main = "",lwd=2,pch=15,ylim=c(0,4),xaxs="i", yaxs="i",axes = FALSE);

#grid(nx = 20);

axis(side=1, at=seq(1,4),labels=model , lwd.ticks=1,cex.axis=0.7)
axis(side=2,  lwd.ticks=1,cex.axis=0.7,las=1)

#text(seq(1,4), par("usr")[3]-0.25, 
#     srt = 60, adj= 1, xpd = TRUE,
#     labels = model, cex=0.65)

title(ylab="Time(sec/scene)", line=2, cex.lab=0.8, family="Helvetica")

abline(h=0,col="black",lwd=2)

legend(x=-25,y=-0.8,legend=model_names,col='black',pch=16,bty="n",xpd=NA,cex = 0.9)

dev.off()
