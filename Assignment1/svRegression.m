(* ::Package:: *)

function [model] = svRegression(X,y, epsilon)
% Solve epsilon incensitive SVM problem through linear program
[n,d] = size(X);
b = [epsilon*ones(n,1);epsilon*ones(n,1);zeros(2*n,1)]+[y;-y;zeros(2*n,1)];
A = [X, ones(n,1),-eye(n), zeros(n,n);-X,-ones(n,1),zeros(n,n),-eye(n);zeros(2*n,2),-eye(2*n)];
f = [0;0;ones(2*n,1)];
w = linprog(f,A,b);
model.w = w(1)
model.beta = w(2)

model.predict = @predict;

end

function [yhat] = predict(model,Xhat)
w = model.w;
beta = model.beta;
yhat = Xhat*w+beta;
end
