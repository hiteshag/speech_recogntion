function [C, dC, direction] = tsne_grad(x, P, const)
%TSNE_GRAD Compute t-SNE gradient and cost function
%
%   [C, dC, direction] = tsne_grad(x, P, const)
%
% Compute t-SNE gradient dC and cost function C.
%
%
% (C) Laurens van der Maaten, 2010
% University of California, San Diego


    % Decode solution
    n = size(P, 1);
    no_dims = length(x) / n;
    Y = reshape(x, [n no_dims]);

    % Compute joint probability that point i and j are neighbors
    sum_Y = sum(Y .^ 2, 2);                                                         % precomputation for pairwise distances
    num = 1 ./ (1 + bsxfun(@plus, sum_Y, bsxfun(@plus, sum_Y', (-2 * Y) * Y')));    % Student-t distribution
    num(1:n + 1:end) = 0;                                                           % remove diagonal
    Q = max(num ./ sum(num(:)), realmin('single'));                                 % normalize to get probabilities
    
    % Compute cost function
    C = const - sum(P(:) .* log(Q(:)));
        
    % Only compute gradient when required
    if nargout > 1
        LP = P .* num; LP = diag(sum(LP, 1)) - LP;
        LQ = Q .* num; LQ = diag(sum(LQ, 1)) - LQ;
        dC = (LP - LQ) * Y; 
        dC = 4 * dC(:);
    end
    
    % Only compute search direction if desired
    if nargout > 2
        direction = (LP \ (LQ * Y)) - Y;
        direction = direction(:);
    end
    