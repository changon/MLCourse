function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(theta'*X')';
%J = (1/m*(-y'*log(h) - ((1-y)'*log(1-h)))) + lambda/(2*m)*(sum(theta.^2));
%grad = ((1/m)*((h-y)'*X)') + lambda/m*theta ;
% we now have the product of  (theta'(X'))'

J = (1/m*(-y'*log(h) - ((1-y)'*log(1-h)))) + lambda/(2*m)*(sum(theta(2:size(X,2)).^2)); %2:size(X,2) excludes theta 0
grad = ((1/m)*((h-y)'*X)') + lambda/m*theta;

%naut_J = (1/m*(-y'*log(h) - ((1-y)'*log(1-h)))) ; %+ lambda/(2*m)*(sum(theta(2:size(X,2),:).^2));
naut_grad = ((1/m)*((h-y)'*X)') ;%+ lambda/m*theta(1:size(X,2),:);

%gradient descent has been performed on theta 0, now, the first column of reg theta naut will be deleted and the product will be appended to the regularized learned parameters 

grad(1) = naut_grad(1); % assign correct gradient value that does not regularize theta(0)

% =============================================================

end
