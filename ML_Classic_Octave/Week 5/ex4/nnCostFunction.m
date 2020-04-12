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
 X = [ones(size(X,1),1) X];         
% You need to return the following variables correctly 
J = 0;
grad = 0 ;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
for i=1:m
  p = zeros(num_labels,1);
  if y(i)==0 
    p(10,1)=1;
  else 
  p(y(i),1) = 1 ; 
  end  
  q = X(i,:);
  a1 = q' ;
  z2 = Theta1*a1 ;
  a2 = sigmoid(z2);
  a2 = [1 ; a2] ;
 % q = [ones(size(q,1),1) q];
  z3 = Theta2*a2;
  a3 = sigmoid(z3);
  J = J + sum( -p.*log(a3) -(1-p).*log(1-a3)); 
  del3 = a3 - p ;
  Theta2_grad = Theta2_grad + del3*a2';
 del2 = Theta2'*del3.*a2.*(1-a2) ; 
 del2 = del2(2:end) ;
% a = a(2:end) ;
 Theta1_grad = Theta1_grad + del2*a1';
endfor
J = J/m ;
a = Theta1; 
b = Theta2;
a(:,1) = 0;
b(:,1) = 0;
J = J + ( sum(sum(a.^2))*lambda/2 + sum(sum(b.^2))*lambda/2 )/m ;
% grad = grad + lambda*(sum(sum(a)) + sum(sum(b))) ;
% grad = grad/m ;
Theta1_grad = Theta1_grad + lambda*a ;
Theta2_grad = Theta2_grad + lambda*b ;
grad = [Theta1_grad(:) ; Theta2_grad(:)];
grad = grad/m ;
end
