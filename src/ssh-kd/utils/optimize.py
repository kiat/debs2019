import math
import numpy as np
import tensorflow as tf
from datetime import datetime

from utils.data_prep import train_validation

def optimize(
    num_iterations,
    train_batch_size,
    input_train,
    output_train,
    session,
    x,
    y_true,
    optimizer,
    accuracy,
    cost,
    prob
):

    # Start time
    start_time = datetime.now()

    # input and validation
    n = len(input_train)
    len_validation_from = n - int(n / 7)

    input_train, output_train, val_input, val_output = train_validation(
        input_train, output_train, len_validation_from
    )

    trian_len = len(input_train)
    val_len = len(val_input)

    # Accuracy and cost lists
    train_loss = []
    val_loss = []

    train_accu = []
    val_accu = []

    num_of_batches = math.ceil(trian_len / train_batch_size)

    # Converting the input into tensors
    x_batch1 = tf.convert_to_tensor(input_train)
    y_true_batch1 = tf.convert_to_tensor(output_train)

    # Creating the input_queue
    input_queue = tf.train.slice_input_producer([x_batch1, y_true_batch1])

    # Slicing the image
    sliced_x = input_queue[0]
    sliced_y = input_queue[1]

    # Batching the queue
    x_batch2, y_true_batch2 = tf.train.batch(
        [sliced_x, sliced_y],
        batch_size=train_batch_size,
        allow_smaller_final_batch=True,
    )

    # Coordinating te multi threaded function
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=session)

    for i in range(num_of_batches * num_iterations):

        x_batch, y_true_batch = session.run([x_batch2, y_true_batch2])

        # Put the batch in the dict with the proper names
        feed_dict_train = {x: x_batch, y_true: y_true_batch, prob: 0.8}

        # Run the optimizer using this batch of the training data
        session.run(optimizer, feed_dict=feed_dict_train)

        # printing status for every 10 iterations
        if i % num_of_batches == 0:

            count = 0
            val_acc = 0
            val_cost = 0

            for j in range(int(val_len / 100)):
                val_acc = val_acc + session.run(
                    accuracy,
                    feed_dict={
                        x: np.array(
                            val_input[j * 100 : (j + 1) * 100], dtype=np.float32
                        ),
                        y_true: np.array(
                            val_output[j * 100 : (j + 1) * 100], dtype=np.float32
                        ),
                    }
                )
                val_cost = val_cost + session.run(
                    cost,
                    feed_dict={
                        x: np.array(
                            val_input[j * 100 : (j + 1) * 100], dtype=np.float32
                        ),
                        y_true: np.array(
                            val_output[j * 100 : (j + 1) * 100], dtype=np.float32
                        ),
                    },
                )
                count += 1

            val_acc = val_acc / count
            val_cost = val_cost / count

            val_accu.append(val_acc)
            val_loss.append(val_cost)

            # Calculating the train accuracy
            count = 0
            train_acc = 0
            train_cost = 0

            for j in range(int(trian_len / 100)):
                train_acc = train_acc + session.run(
                    accuracy,
                    feed_dict={
                        x: np.array(
                            input_train[j * 100 : (j + 1) * 100], dtype=np.float32
                        ),
                        y_true: np.array(
                            output_train[j * 100 : (j + 1) * 100], dtype=np.float32
                        ),
                        prob:0.8
                    },
                )
                train_cost = train_cost + session.run(
                    cost,
                    feed_dict={
                        x: np.array(
                            input_train[j * 100 : (j + 1) * 100], dtype=np.float32
                        ),
                        y_true: np.array(
                            output_train[j * 100 : (j + 1) * 100], dtype=np.float32
                        ),
                        prob:0.8
                    },
                )
                count += 1

            train_acc = train_acc / count
            train_cost = train_cost / count

            train_accu.append(train_acc)
            train_loss.append(train_cost)

            print("---------")
            print(
                "Optimization Epochs: {0:>6}, Training Accuracy: {1:6.1%}, validation Accuracy: {2:6.1%}, training cost: {3}, val_cost: {4}".format(
                    (i / num_of_batches) + 1, train_acc, val_acc, train_cost, val_cost
                )
            )

    coord.request_stop()
    coord.join(threads)

    # Ending time
    end_time = datetime.now()

    print("Time usage: {}".format(end_time - start_time))
    return train_accu, val_accu, train_loss, val_loss


def print_test_accuracy(
    test_input, test_output, session, y_true, y_pred_cls, x, show_confusion_matrix=False
):

    # number of images in the test -set
    num_test = len(test_input)

    # creating an empty array
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Starting index
    i = 0

    test_batch_size = 64

    while i < num_test:
        # J is the ending index
        j = min(i + test_batch_size, num_test)

        # get the images
        images = test_input[i:j]

        # Get the assiciated labels
        labels = test_output[i:j]

        # Feed the dict with the images and labels
        feed_dict = {x: images, y_true: labels}

        # Calculate the predicated class using TensorFlow
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        i = j

    cls_true = [np.argmax(i) for i in test_output]
    cls_true = np.array(cls_true)

    correct = cls_true == cls_pred
    correct_sum = correct.sum()

    acc = float(correct_sum) / num_test

    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"

    print(msg.format(acc, correct_sum, num_test))

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        # Visualization().plot_confusion_matrix(cls_pred, cls_true)