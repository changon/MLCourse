function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C_vec = [.01,.03, .1, .3, 1, 3, 10, 30];
sigma_vec = [.01,.03, .1, .3, 1, 3, 10, 30];

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
results=[];
for i = 1:length(C_vec)
  currentc = C_vec(i);
  for j = 1:length(sigma_vec)
    currentsigma = sigma_vec(j);
    model = svmTrain(X, y, currentc, @(x1, x2) gaussianKernel(x1, x2, currentsigma));
    predictions = svmPredict(model, Xval);
    err = mean(double(predictions ~= yval));
    results = [results; currentc, currentsigma, err]; %https://github.com/TomLous/coursera-machine-learning/blob/master/mlclass-ex6/dataset3Params.m
  end;
end;
size(results)
[e, index] = min(results(:,3));
C = results(index, 1);
sigma = results(index, 2);

% =========================================================================

end
