from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    num_train = X.shape[0]

    ##########################
    # ***** Solution 1 ***** #
    ##########################
    # for i in range(num_train):
    #     scores = X[i].dot(W)

    #     # Normalization trick to avoid numerical instability
    #     scores -= np.max(scores)

    #     prob_numerator = np.exp(scores)
    #     prob_denominator = np.sum(np.exp(scores))

    #     # Calculate d_numerator_yi_d_W
    #     temp_dW = np.zeros_like(W)
    #     temp_dW[:,y[i]] += X[i,:]
    #     temp_dW[:,np.argmax(scores)] -= X[i,:]
    #     d_numerator_yi_d_W = prob_numerator[y[i]] * temp_dW

    #     # Calculate d_denominator_d_W
    #     d_denominator_d_W = np.zeros_like(W)
    #     for j in range(num_classes):
    #         temp_dW = np.zeros_like(W)
    #         temp_dW[:,j] += X[i,:]
    #         temp_dW[:,np.argmax(scores)] -= X[i,:]
    #         d_denominator_d_W += prob_numerator[j] * temp_dW

    #     prob = prob_numerator/prob_denominator

    #     # Calculate d_prob_yi_d_numerator_yi and d_prob_yi_d_denominator
    #     d_prob_yi_d_numerator_yi = 1 / prob_denominator
    #     d_prob_yi_d_denominator = -prob[y[i]] / prob_denominator

    #     cross_entropy = -np.log(prob[y[i]])

    #     # Calculate d_cross_entropy_d_prob_yi
    #     d_cross_entropy_d_prob_yi = -1 / prob[y[i]]

    #     loss += cross_entropy
    #     dW += d_cross_entropy_d_prob_yi * (d_prob_yi_d_numerator_yi * d_numerator_yi_d_W  + d_prob_yi_d_denominator * d_denominator_d_W)
    ##########################
    # *** Solution 1 end *** #
    ##########################


    ##########################
    # ***** Solution 2 ***** #
    ##########################
    for i in range(num_train):
        # Compute vector of scores
        scores = X[i].dot(W)

        # Normalization trick to avoid numerical instability
        log_c = np.max(scores)
        scores -= log_c
        scores_exp = np.exp(scores)
        scores_exp_sum = np.sum(scores_exp)

        # Compute loss
        loss = loss - scores[y[i]] + np.log(scores_exp_sum)

        # Compute gradient
        for j in range(num_classes):
            p = scores_exp[j] / scores_exp_sum
            dW[:, j] += (p-(j == y[i])) * X[i, :]
    ##########################
    # *** Solution 2 end *** #
    ##########################

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    num_train = X.shape[0]

    ##########################
    # ***** Solution 1 ***** #
    ##########################
    # scores = np.dot(X, W)
    # max_scores = np.max(scores, axis=1).reshape(-1,1)
    # argmax_scores = np.argmax(scores, axis=1)
    # scores -= max_scores

    # prob_numerator = np.exp(scores)
    # prob_denominator = np.sum(np.exp(scores), axis=1)
    # prob_matrix = prob_numerator / prob_denominator.reshape(-1,1)
    # prob = prob_numerator[xrange(num_train),y]/prob_denominator

    # cross_entropy = -np.log(prob)
    # loss = np.sum(cross_entropy) / num_train
    # loss += 0.5 * reg * np.sum(W * W)

    # binary = np.zeros((num_train, num_classes))
    # binary[range(num_train),y] -= 1
    # binary[range(num_train),argmax_scores] += 1

    # prob_matrix_revised = prob_matrix
    # prob_matrix_revised[range(num_train),argmax_scores] -= 1
    # dW += np.dot(X.T, binary+prob_matrix_revised)
    # dW /= num_train
    # dW += reg*W
    ##########################
    # *** Solution 1 end *** #
    ##########################


    ##########################
    # ***** Solution 2 ***** #
    ##########################
    # Compute scores
    scores = np.dot(X, W)

    # Normalization trick to avoid numerical instability
    scores -= np.max(scores, axis=1).reshape(-1,1)
    scores_exp = np.exp(scores)
    scores_exp_sum = np.sum(scores_exp, axis=1)
    scores_correct = scores[range(num_train),y]

    # Compute loss
    losses = -scores_correct + np.log(scores_exp_sum)
    loss += np.mean(losses)

    # Compute gradient
    prob_matrix = scores_exp / scores_exp_sum.reshape(-1,1)
    ind = np.zeros(prob_matrix.shape)
    ind[range(num_train),y] = 1
    dW = np.dot( X.T , (prob_matrix-ind) )
    dW /= num_train

    # Regularization
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg*W
    ##########################
    # *** Solution 2 end *** #
    ##########################
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
