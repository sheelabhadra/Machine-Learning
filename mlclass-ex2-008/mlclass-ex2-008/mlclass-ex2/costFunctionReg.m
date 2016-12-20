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

h = sigmoid(X*theta);  % function for logistic regression

for i = 1:m
    J = J + (-((y(i))*log(h(i)))-((1-y(i))*log(1-h(i))));  % cost-function
end

J = J/m ;

reg = 0;

for j = 2:size(X,2)
    reg = reg + ((lambda)*theta(j)*theta(j))/(2*m);
end
 
J = J + reg ;

for j = 1:size(theta)
    for i = 1:m
    grad(j) = grad(j) + (h(i) - y(i))*X(i,j);
    end
end

grad(1) = grad(1)/m ;
for j =2:size(theta)
    grad(j) = (grad(j) + (lambda)*theta(j))/m ;
end

% =============================================================

end
