# Fire and Smoke Segmentation Using Active Learning Methods

**NOTE: Classification and Segmentation phases have different software requirements. Thus, it is recommended for the user to create two different virtual environments (ex: Anaconda).**


-----------------------------------------------------
Work available at: https://doi.org/10.3390/rs15174136

## Abstract
This work proposes an active learning (AL) methodology to create models for the segmentation of fire and smoke in video images. With this model, a model learns in an incremental manner over several AL rounds. Initially, the model is trained in a given subset of samples, and in each AL round, the model selects the most informative samples to be added to the training set in the next training session. Our approach is based on a decomposition of the task in an AL classification phase, followed by an attention-based segmentation phase with class activation mapping on the learned classifiers. The use of AL in classification and segmentation tasks resulted in a 2% improvement in accuracy and mean intersection over union. More importantly, we showed that the approach using AL achieved results similar to non-AL with fewer labeled data samples.

## Keywords
classification; segmentation; deep learning; active learning; convolutional neural networks 


![CEAL2](https://github.com/trmarto/fire-smoke-AL/assets/74827101/2e2d2ef5-c544-4d48-8d93-4b930b1cc189)


