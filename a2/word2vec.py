#!/usr/bin/env python

import argparse
import numpy as np
import random

from utils.gradcheck import gradcheck_naive, grad_tests_softmax, grad_tests_negsamp
from utils.utils import normalizeRows, softmax


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    Arguments:
    x -- A scalar or numpy array.
    Return:
    s -- sigmoid(x)
    """

    ### YOUR CODE HERE (~1 Line)

    # Simple sigmoid function implementation.
    s = 1 / (1 + np.exp(-x))

    ### END YOUR CODE

    return s


def naiveSoftmaxLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset
):
    """ Naive Softmax loss & gradient function for word2vec models

    Implement the naive softmax loss and gradients between a center word's 
    embedding and an outside word's embedding. This will be the building block
    for our word2vec models. For those unfamiliar with numpy notation, note 
    that a numpy ndarray with a shape of (x, ) is a one-dimensional array, which
    you can effectively treat as a vector with length x.

    Arguments:
    centerWordVec -- numpy ndarray, center word's embedding
                    in shape (word vector length, )
                    (v_c in the pdf handout)
    outsideWordIdx -- integer, the index of the outside word
                    (o of u_o in the pdf handout)
    outsideVectors -- outside vectors is
                    in shape (num words in vocab, word vector length) 
                    for all words in vocab (tranpose of U in the pdf handout)
    dataset -- needed for negative sampling, unused here.

    Return:
    loss -- naive softmax loss
    gradCenterVec -- the gradient with respect to the center word vector
                     in shape (word vector length, )
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length) 
                    (dJ / dU)
    """

    ### YOUR CODE HERE (~6-8 Lines)

    ### Please use the provided softmax function (imported earlier in this file)
    ### This numerically stable implementation helps you avoid issues pertaining
    ### to integer overflow. 
    """
    Computation of the loss:
    - Compuation of the scaled dot product between the outside word(s) and the center word(s).
      Denoted as scores, which is unnormalized.
    - Computation of the softmax of scores, which creates a probability distribution of the
      outside word(s) given the center word(s). Denoted as y_hat.
    - Computation of the scalar naive softmax loss, which is simply the cross entropy loss
      (negative log likelihood) betweeen the ground truth y and the predicted y_hat. 
    """
    scores = outsideVectors.dot(centerWordVec)
    y_hat = softmax(scores)
    loss = -np.log(y_hat[outsideWordIdx])

    """
    Computation of the gradients of the loss wrt. the center word(s):
    - The chain rule of the derivative of the loss wrt. the scores, and the derivative 
      of the scores wrt. the center word(s).
    - Without complex computation, the derivative of the loss wrt. the scores is simply
      the predicted truth - the ground truth (y_hat - y)
    - The derivative of the scores wrt. the center word(s) is the outside word(s)
    """
    # ground truth y, which have to be the same shape as the prdicted truth, y_hat.
    y = np.zeros_like(y_hat)
    y[outsideWordIdx] = 1
    gradCenterVec = (y_hat - y).dot(outsideVectors)

    """
    Computation of the gradients of the loss wrt. the outside word(s):
    - Similar to the computation of the gradients of the center word(s), except we can 
      compute the outer product of (y_hat - y) and centerWordVec, which is the derivative 
      of the scores wrt. the outside word(s).
    - The gradients will form a jacobian matrix, where each element is the product of the 
      corresponding elements in the input vectors.
    - We need to compute this jacobian because the matrix that holds each gradient will have 
      to be of the same shape as the outsideVectors, and will have the gradients of the loss
      wrt. to each outside word.
    """
    gradOutsideVecs = np.outer(y_hat - y, centerWordVec)

    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs


def getNegativeSamples(outsideWordIdx, dataset, K):
    """ Samples K indexes which are not the outsideWordIdx """

    negSampleWordIndices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == outsideWordIdx:
            newidx = dataset.sampleTokenIdx()
        negSampleWordIndices[k] = newidx
    return negSampleWordIndices


def negSamplingLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset,
    K=10
):
    """ Negative sampling loss function for word2vec models

    Implement the negative sampling loss and gradients for a centerWordVec
    and a outsideWordIdx word vector as a building block for word2vec
    models. K is the number of negative samples to take.

    Note: The same word may be negatively sampled multiple times. For
    example if an outside word is sampled twice, you shall have to
    double count the gradient with respect to this word. Thrice if
    it was sampled three times, and so forth.

    Arguments/Return Specifications: same as naiveSoftmaxLossAndGradient
    """

    # Negative sampling of words is done for you. Do not modify this if you
    # wish to match the autograder and receive points!

    # Obtains the k negative samples in the set.
    negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
    
    # Creation of the list of indices containing the index of the outside word and the 
    # indices of the negative samples.
    indices = [outsideWordIdx] + negSampleWordIndices

    ### YOUR CODE HERE (~10 Lines)
    ### Please use your implementation of sigmoid in here.

    """
    Negative sampling is used to address the computational inefficiency of the softmax
    function ofr word2vec, by updating the weights of the target word and a small number of 
    randomly chosen words (negative samples), rather than all of the weights for every word.

    We will:
        - Get a center and outside word pair, denote this pair as the positive sample.
        - Randomly select k negative samples (words) from the dataset. Chosen based on their
          frequency in the set. 
        - Updating of weights for the outside word and the k negative samples, increasing 
          the similarity between the center word and the outside word, while decreasing 
          the similarity betwen the center word and the negative samples.
        - Do same process for all the center word, outside word pairs.
    """
    # We will multiply where same words are involved, not recalculating.
    un, idx, n_reps = np.unique(indices, return_index=True, return_counts=True)
    U_concat = outsideVectors[un]
    
    # For convenience
    n_reps[idx==0] *= -1
    U_concat[idx!=0] *= -1
    S = sigmoid(centerWordVec @ U_concat.T)
    
    # Find loss and derivatives w.r.t. v_c, U
    loss = -(np.abs(n_reps) * np.log(S)).sum()
    gradCenterVec = np.abs(n_reps) * (1 - S) @ -U_concat
    gradOutsideVecs = np.zeros_like(outsideVectors)
    gradOutsideVecs[un] = n_reps[:, None] * np.outer(1 - S, centerWordVec)

    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs


def skipgram(currentCenterWord, windowSize, outsideWords, word2Ind,
             centerWordVectors, outsideVectors, dataset,
             word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentCenterWord -- a string of the current center word
    windowSize -- integer, context window size
    outsideWords -- list of no more than 2*windowSize strings, the outside words
    word2Ind -- a dictionary that maps words to their indices in
              the word vector list
    centerWordVectors -- center word vectors (as rows) is in shape 
                        (num words in vocab, word vector length) 
                        for all words in vocab (V in pdf handout)
    outsideVectors -- outside vectors is in shape 
                        (num words in vocab, word vector length) 
                        for all words in vocab (transpose of U in the pdf handout)
    word2vecLossAndGradient -- the loss and gradient function for
                               a prediction vector given the outsideWordIdx
                               word vectors, could be one of the two
                               loss functions you implemented above.

    Return:
    loss -- the loss function value for the skip-gram model
            (J in the pdf handout)
    gradCenterVec -- the gradient with respect to the center word vector
                     in shape (num words in vocab, word vector length)
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length) 
                    (dJ / dU)
    """

    loss = 0.0
    gradCenterVecs = np.zeros(centerWordVectors.shape)
    gradOutsideVectors = np.zeros(outsideVectors.shape)

    ### YOUR CODE HERE (~8 Lines)

    """
    We will:
        - Map the given center word to its respective index in word2Ind.
        - Use that index to get the embedded word vector that corresponds to that given center word.
        - For each center word, outside word pair (within the specified window size):
            - compute the loss and gradients of the center and outside word pair
            - add the losses and gradients up at each iteration (for each pair).
    """
    centerWordIdx = word2Ind[currentCenterWord]
    centerWordVec = centerWordVectors[centerWordIdx]
    # loop that adds the loss and gradients for each center word, outside word pair.
    for outsideWord in outsideWords:
        loss_iter, gradCenterVec_iter, gradOutsideVec_iter = word2vecLossAndGradient(
            centerWordVec, word2Ind[outsideWord], outsideVectors, dataset)
        loss += loss_iter
        # Add the calculated gradient to its specified index which correspond to the center word's index,
        # since at each iteration, the shape of those gradients will only be the word vector's length.
        gradCenterVecs[centerWordIdx] += gradCenterVec_iter
        # Add the calcuated gradient to the gradients of the outside vectors.
        gradOutsideVectors += gradOutsideVec_iter
    ### END YOUR CODE
    
    return loss, gradCenterVecs, gradOutsideVectors


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, word2Ind, wordVectors, dataset,
                         windowSize,
                         word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    batchsize = 50
    loss = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    centerWordVectors = wordVectors[:int(N/2),:]
    outsideVectors = wordVectors[int(N/2):,:]
    for i in range(batchsize):
        windowSize1 = random.randint(1, windowSize)
        centerWord, context = dataset.getRandomContext(windowSize1)

        c, gin, gout = word2vecModel(
            centerWord, windowSize1, context, word2Ind, centerWordVectors,
            outsideVectors, dataset, word2vecLossAndGradient
        )
        loss += c / batchsize
        grad[:int(N/2), :] += gin / batchsize
        grad[int(N/2):, :] += gout / batchsize

    return loss, grad

def test_sigmoid():
    """ Test sigmoid function """
    print("=== Sanity check for sigmoid ===")
    assert sigmoid(0) == 0.5
    assert np.allclose(sigmoid(np.array([0])), np.array([0.5]))
    assert np.allclose(sigmoid(np.array([1,2,3])), np.array([0.73105858, 0.88079708, 0.95257413]))
    print("Tests for sigmoid passed!")

def getDummyObjects():
    """ Helper method for naiveSoftmaxLossAndGradient and negSamplingLossAndGradient tests """

    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in range(2*C)]

    dataset = type('dummy', (), {})()
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])

    return dataset, dummy_vectors, dummy_tokens

def test_naiveSoftmaxLossAndGradient():
    """ Test naiveSoftmaxLossAndGradient """
    dataset, dummy_vectors, dummy_tokens = getDummyObjects()

    print("==== Gradient check for naiveSoftmaxLossAndGradient ====")
    def temp(vec):
        loss, gradCenterVec, gradOutsideVecs = naiveSoftmaxLossAndGradient(vec, 1, dummy_vectors, dataset)
        return loss, gradCenterVec
    gradcheck_naive(temp, np.random.randn(3), "naiveSoftmaxLossAndGradient gradCenterVec")

    centerVec = np.random.randn(3)
    def temp(vec):
        loss, gradCenterVec, gradOutsideVecs = naiveSoftmaxLossAndGradient(centerVec, 1, vec, dataset)
        return loss, gradOutsideVecs
    gradcheck_naive(temp, dummy_vectors, "naiveSoftmaxLossAndGradient gradOutsideVecs")

def test_negSamplingLossAndGradient():
    """ Test negSamplingLossAndGradient """
    dataset, dummy_vectors, dummy_tokens = getDummyObjects()

    print("==== Gradient check for negSamplingLossAndGradient ====")
    def temp(vec):
        loss, gradCenterVec, gradOutsideVecs = negSamplingLossAndGradient(vec, 1, dummy_vectors, dataset)
        return loss, gradCenterVec
    gradcheck_naive(temp, np.random.randn(3), "negSamplingLossAndGradient gradCenterVec")

    centerVec = np.random.randn(3)
    def temp(vec):
        loss, gradCenterVec, gradOutsideVecs = negSamplingLossAndGradient(centerVec, 1, vec, dataset)
        return loss, gradOutsideVecs
    gradcheck_naive(temp, dummy_vectors, "negSamplingLossAndGradient gradOutsideVecs")

def test_skipgram():
    """ Test skip-gram with naiveSoftmaxLossAndGradient """
    dataset, dummy_vectors, dummy_tokens = getDummyObjects()

    print("==== Gradient check for skip-gram with naiveSoftmaxLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, naiveSoftmaxLossAndGradient),
        dummy_vectors, "naiveSoftmaxLossAndGradient Gradient")
    grad_tests_softmax(skipgram, dummy_tokens, dummy_vectors, dataset)

    print("==== Gradient check for skip-gram with negSamplingLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingLossAndGradient),
        dummy_vectors, "negSamplingLossAndGradient Gradient")
    grad_tests_negsamp(skipgram, dummy_tokens, dummy_vectors, dataset, negSamplingLossAndGradient)

def test_word2vec():
    """ Test the two word2vec implementations, before running on Stanford Sentiment Treebank """
    test_sigmoid()
    test_naiveSoftmaxLossAndGradient()
    test_negSamplingLossAndGradient()
    test_skipgram()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test your implementations.')
    parser.add_argument('function', nargs='?', type=str, default='all',
                        help='Name of the function you would like to test.')

    args = parser.parse_args()
    if args.function == 'sigmoid':
        test_sigmoid()
    elif args.function == 'naiveSoftmaxLossAndGradient':
        test_naiveSoftmaxLossAndGradient()
    elif args.function == 'negSamplingLossAndGradient':
        test_negSamplingLossAndGradient()
    elif args.function == 'skipgram':
        test_skipgram()
    elif args.function == 'all':
        test_word2vec()
