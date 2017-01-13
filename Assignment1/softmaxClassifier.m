function [model,funEvals,backtrackEvals] = softmaxClassifier(X,y)
% Regression using softmax probability model

% Compute sizes
[n,d] = size(X);
k = max(y);

% Matrix representing the Kronecker delta \delta_{y^i, c}
Y = repmat(y,[1 k]);
C = repmat(1:k,[n 1]);
deltaCY = C==Y;

% Calculating the weights W
softmaxLoss0 = @(W,X,y) softmaxLoss(W,X,y,deltaCY);

W = zeros(d,k);
[W,~,funEvals,backtrackEvals] = findMin(softmaxLoss0,W(:),500,1,X,y);     
W = reshape(W,[d k]);

model.W = W;
model.predict = @predict;
end

function [yhat,probTable] = predict(model,X)
W = model.W; XW = X*W; k = size(XW,2);
[~,yhat] = max(X*W,[],2);       % maximum probability
probTable = exp(XW)./repmat(sum(exp(XW),2),[1 k]);
% sum(probTable,2) should add up to 1 for every row (safety check)
end

function [f,g] = softmaxLoss(W,X,y,deltaCY)

[~,d] = size(X); k = max(y);
W = reshape(W,[d k]);           % reshape W to vectorize this part
XW = (X*W);                     % nxk matrix

f = -trace(XW*deltaCY') + sum(log(sum(exp(XW),2)));      % Loss function
g = -X'*deltaCY + X'*diag(1./sum(exp(XW),2))*exp(XW);    % Gradient matrix
g = g(:);   % Reshape g into a vector

end