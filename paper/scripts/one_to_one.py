# imports
import matplotlib.pyplot as plt

# Change the size of the image here
plt.rcParams["figure.figsize"] = (4,4)

# Train and validation accuracy
train_acc = [4.0,30.1,42.3,54.6,61.8,67.6,72.5,75.9,78.7,81.7,84.7,87.2]
val_acc = [3.9,32.3,47.2,62.9,69.2,74.9,77.9,81.2,82.5,85.4,88.3,92.7]

# Train and validation loss
train_loss = [3.347,2.224,1.742,1.408,1.1798,0.98703,0.8687,0.7552,0.6347,0.5813,0.5345,0.4935]
val_loss = [3.339,2.1006,1.5959,1.2513,1.0288,0.88,0.7654,0.6983,0.6212,0.5836,0.5516,0.4953]

test_accu = 94.2



fig = plt.figure()

# lines
plt.plot(range(1,13),train_acc,label='Training Accuracy')
plt.plot(range(1,13),val_acc,label='Validation Accuracy')

# axis labels
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

# title
plt.title('Object-net')

# ticks and legend
plt.xticks(range(1,13))
plt.legend()

# saving
fig.savefig("../images/accuracy.png")

# clearing the plot after saving
plt.clf()

fig = plt.figure()
plt.plot(range(1,13),train_loss,label='Training loss')
plt.plot(range(1,13),val_loss, label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Object-net')
plt.xticks(range(1,13))
plt.legend()
fig.savefig("../images/loss.png")
plt.clf()
