%% load utils and MNIST data
clc; clear; tic;
addpath('./utils');
[train, test] = DataPrep('./data');
toc;

%% TODO
% ...

%% a sample usage of ShowModel
mu = train.images(:, 1:8);
pi = rand(1, 8);
pi = pi/sum(pi);
ShowModel(mu, pi, 2, 4, 1:8);