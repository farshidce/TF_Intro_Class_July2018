def predict_cls(images, labels, cls_true):
    # Number of images.
    num_images = len(images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)
        
        if images[i:j,:].shape[0] != batch_size:
            break

        #print("images[i:j, :] shape: ",images[i:j, :].shape)
        #print("labels[i:j, :] shape: ",labels[i:j, :].shape)

        # Create a feed-dict with the images and labels
        # between index i and j.
        feed_dict = {x: images[i:j, :],
                     y_true: labels[i:j, :]}

        # Calculate the predicted class using TensorFlow.
        try:
            cls_pred[i:j] = sess.run(y_pred_cls, feed_dict=feed_dict)
        except tf.errors.OutOfRangeError or ValueError:
            pass

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Create a boolean array whether each image is correctly classified.
    assert cls_true.shape == cls_pred.shape
    correct = (cls_true == cls_pred)

    return correct, cls_pred