(* ::Package:: *)

function [model] = RobustRegression(X,y)
% Solve L1-norm problem through linear program  
[n,d] = size(X);
b = zeros(2*n,1);
beq = y(:);
A = [zeros(n,2), eye(n), -eye(n);zeros(n,2) -eye(n) -eye(n)];
Aeq = [X, ones(n,1), eye(n), zeros(n,n)];
f = [0;0;zeros(n,1); ones(n,1)];
w = linprog(f,A,b,Aeq,beq);
model.w = w(1);
model.beta = w(2);

model.predict = @predict;

end

function [yhat] = predict(model,Xhat)
w = model.w;
beta = model.beta;
yhat = Xhat*w+beta;
end
