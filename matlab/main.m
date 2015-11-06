clc; clear; tic;
addpath('./utils');
[train, test] = DataPrep('./data');
toc;

[mu, pi_, z] = EM(train, test);

loglikelihood(train.images.', mu, pi_, z)

loglikelihood(test.images.', mu, pi_, z)

ShowModel(mu.', pi_, 5, 8, 1:40);