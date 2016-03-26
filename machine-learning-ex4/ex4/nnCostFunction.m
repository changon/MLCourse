function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%Forward propagation-----------------------
X =[ones(m,1) X];
%bias unit add +. 1's
z1 = (Theta1*X'); %t1 25X401, X' 401X5000 => 25X5000
a2 = sigmoid(z1)';
a2 = [ones(size(X,1), 1) a2]; % 5000X26
%run theta1 through cost funct.
z2 = Theta2*a2'; % t2 10X26, h1' 26X5000 => 10 X 5000 
h2 = sigmoid(z2');  

yy = zeros(num_labels, m);
for i = 1:m
  yy(y(i), i) = 1;
end;

% relabel vector as logic array with discrete binary vals 
%J = (1/m*(sum(-yy.*log(h2)) - sum((1-yy).*log(1-h2)))); %2:size(X,2) excludes theta 0
J = (1/m) * sum ( sum (  (-yy') .* log(h2)  -  (1-yy') .* log(1-h2) ));

%Regularization phase------------------------
t_one = Theta1(:,2:size (Theta1,2)); %DO NOT regularize theta 0 or the bias unit 
t_two = Theta2(:,2:size (Theta2,2)); %dim of theta2 is 26, 2:26 in order to avoid this bias unit which is 1
reg = (lambda / (2*m))*(sum(sum(t_one.^2)) + sum(sum(t_two.^2)));
J += reg;

%Backpropagation-----------------------------

for i = 1:m
  %FP, pass by pass 
  a_1 = X(i,:); 
  z_1 = (Theta1 * a_1');  %25x1
  a_2 = sigmoid(z_1); %26x1
  a_2 = [1; a_2]; 
  z_2 = Theta2 * a_2; %10x1
  a_3 = sigmoid(z_2); %1x10
	fprintf(" %d is size of z1, %d is size of a2 ", size(a_1,1), size(a_1,2)); 
	fprintf(" %d is size of a1, %d is size of z1 ", size(z_1,1), size(z_1,2));  
	fprintf(" %d is size of z2, %d is size of a2 ", size(a_2,1), size(a_2,2));  
	fprintf(" %d is size of a2, %d is size of a2 ", size(z_2,1), size(z_2,2));  
	fprintf(" %d is size of a1, %d is size of a2 ", size(a_1,1), size(a_1,2));    
  %true back propagation 
  z_1 = [1;z_1]; %to account for the bias unit
  fprintf("backprop");
  fprintf(" %d is size of a1, %d is size of a1 ", size(z_1,1), size(z_1,2));  
  
  dd_3 = a_3 - yy(:, i); %denotes lowercase delta layer 3, unit k  10X1
  fprintf(" %d is size of ysize, %d is size of ysize ", size(yy(:,i),1), size(yy(:,i),2));  
  fprintf(" %d is size of a3, %d is size of a3 ", size(a_3,1), size(a_3,2));  
  fprintf(" %d is size of dd_3, %d is size of dd_3 ", size(dd_3,1), size(dd_3,2));  

  dd_2 = (Theta2'*dd_3).*sigmoidGradient(z_1); %26X1.* z_1
  fprintf(" %d is size of ysize, %d is size of ysize ", size(yy(:,i),1), size(yy(:,i),2));  
  fprintf(" %d is size of a3, %d is size of a3 ", size(a_3,1), size(a_3,2));  
  fprintf(" %d is size of dd_2, %d is size of dd_2 ", size(dd_3,1), size(dd_3,2));  

  dd_2 = dd_2 (2:end); %rm bias
  Theta2_grad += dd_3 *a_2';
  Theta1_grad += dd_2 *a_1;
  fprintf(" %d is size of theta2, %d is size of theta2 ", size(Theta2_grad,1), size(Theta2_grad,2));  
  fprintf(" %d is size of theta1, %d is size of theta1 ", size(Theta1_grad,1), size(Theta1_grad,2));  

  %26x1 
end;

Theta2_grad /= m; %/m
Theta1_grad /= m;
% -------------------------------------------------------------

%Regularization of gradients, NN
Theta1_grad(:,2:end) += (lambda/m)*Theta1(:,2:end);
Theta2_grad(:,2:end) += (lambda/m)*Theta2(:,2:end);
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end