# Stable Update of Regression Trees

A method for updating regression trees in a stable manner based on uncertainty-weighted stability regularization.  

## Abstract
Updating machine learning models with new information usually improves their predictive performance, yet, in many applications, it is also desirable to avoid changing the model predictions too much. This property is called stability.In most cases when stability matters, so does explainability. We therefore focus on the stability of an inherently explainable machine learning method, namely regression trees. We aim to use the notion of *empirical stability* and design algorithms for updating regression trees that provide a way to balance between predictability and empirical stability. To achieve this, we propose a regularization method, where data points are weighted based on the uncertainty in the initial model. The balance between predictability and empirical stability can be adjusted through hyperparameters. This regularization method is evaluated in terms of loss and stability and assessed on a broad range of data characteristics. The results show that the proposed update method improves stability while achieving similar or better predictive performance. This shows that it is possible to achieve both predictive and stable results when updating regression trees. 

