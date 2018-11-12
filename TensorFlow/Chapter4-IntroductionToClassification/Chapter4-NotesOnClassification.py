''' The previous chapter dealt with regression, which was about fitting a curve to data.
The best-fit curve was a function that takes as input a data item and assigns it a number.
Creating a machine learning model that instead assigns discrete labels to its inputs is
called classification. It is a supervised learning algorithm for dealing with discrete outputs
(called classes). A classifier may take either continuous or discrete variables as inputs.
If there are only two class labels, then we call this learning algorithm a binary classifier.
Otherwise, it is called a multiclass classifier.
'''

#########################################################################
# The different types of discrete variables, and how to store nominal variables
#########################################################################
'''  Remember that discrete variables can either be ordinal (i.e. that can be ordered; e.g. 2,
4, 6), or nominal (i.e. that cannot be ordered, and therefore are only described by their name;
e.g. apple, orange, pear). A simple approach to represent nominal variables in a dataset is to
assign a number to each label (e.g. [1, 2, 3] for [apple, orange, pear]). However, some
classification models may interpret such labels incorrectly. For example, linear regression
would see 1.5 as halfway between an apple and an orange (which makes no natural sense). The
work-around solution is to create DUMMY VARIABLES for each value of the nominal variance. In
the current example, this would mean that three seperate variables would be created, referring
independently to apples, oranges, and pears. Each variable would hold a value of 0 or 1, depending
on whether the category for that fruit holds true. This is often referred to as ONE-HOT ENCODING.
'''

#########################################################################
# Measuring classification model performance
#########################################################################

# Accuracy
''' One common measure used to assess the performance of a classification model is ACCURACY.
Accuracy is defined as the total number of correct answers, divided by the total number of
answers. This formula gives a crude summary of the overall correctness of the algorithm.
However, it does not give any breakdown of the correct and incorrect results for each label.

    To account for this limitation, a CONFUSION MATRIX gives a more detailed report of the
classifiers success. Specifically, it compares the predicted responses of a model compared
to the actual ones For example, when assessing a binary classifer, a confusion matrix will
be a table showing the true positives and negatives, versus false positives and negatives.
This table is called a confusion matrix because it makes it easy to see how often a model
confuses two classes that it is trying to differentiate.
'''

# Precision and recall
''' Although the definitions of true positives/negatives and false positives/negatives are
useful, their true power comes in their interplay. For example, the ratio of true positives
to total positive examples is called PRECISION. Precision is therefore a score of how likely
a positive prediction is to be correct:

    PRECISION = true positives / (true positives + false positives)

    In contrast, the ratio of true positives to all possible positives is called RECALL. It
measures the ratio of true positives found. In other words, it is a score of how many true-
positives were successfully predicted (that is, "recalled").

    RECALL = true positives / (true positives + false negative)

    Simply put, precision is the proportion of your predictions you got right and recall is
the proportion of the right things you identified in the final set.
'''

# Receiver operating characteristic curves
''' Another way to assess the performance of binary classifiers is through the use of
receiver operating characteristic (ROC) curves. The ROC curve is a plot of trade-offs
between false-positives and true-positives. The x-axis is the measure of false-positive
values, and the y-axis is the measure of true-positive values. These ROC curves can be
compared between models. When two model curves do not intersect, then one method is certainly
better than the other. Good algorithms are way above the baseline (i.e. a positive, diagonal
line through the center of the graph; see ROC curves.png). A quantitative way to compare
classifiers is by measuring the AREA UNDER THE ROC CURVE. If a model has an area-under-curve
(AUC) value higher than 0.9, the it is an excellent classifier. A model that randomly guesses
the output will have an AUC value of about 0.5.
'''