function [Y, Cs, ts] = fast_tsne(X, labels, no_dims, initial_dims, perplexity)
%FAST_TSNE Performs symmetric t-SNE on dataset X
%
%   mappedX = fast_tsne(X, labels, no_dims, initial_dims, perplexity)
%   mappedX = fast_tsne(X, labels, initial_solution, perplexity)
%
% The function performs symmetric t-SNE on the NxD dataset X to reduce its 
% dimensionality to no_dims dimensions (default = 2). The data is 
% preprocessed using PCA, reducing the dimensionality to initial_dims 
% dimensions (default = 30). Alternatively, an initial solution obtained 
% from an other dimensionality reduction technique may be specified in 
% initial_solution. The perplexity of the Gaussian kernel that is employed 
% can be specified through perplexity (default = 30). The labels of the
% data are not used by t-SNE itself, however, they are used to color
% intermediate plots. Please provide an empty labels matrix [] if you
% don't want to plot results during the optimization.
% The low-dimensional data representation is returned in mappedX.
%
%
% (C) Laurens van der Maaten, 2010
% University of California, San Diego


    if ~exist('labels', 'var')
        labels = [];
    end
    if ~exist('no_dims', 'var') || isempty(no_dims)
        no_dims = 2;
    end
    if ~exist('initial_dims', 'var') || isempty(initial_dims)
        initial_dims = 30;
    end
    if ~exist('perplexity', 'var') || isempty(perplexity)
        perplexity = 30;
    end
    
    % Prewhiten and normalize input data
    
    fprintf('Prewhitening and normalizing input data\n');
    X = bsxfun(@minus, X, mean(X, 1));
    covX = X' * X;
    [M, lambda] = eig(covX);
    [foo, ind] = sort(diag(lambda), 'descend');
    if initial_dims > size(M, 2)
        initial_dims = size(M, 2);
    end
    M = M(:,ind(1:initial_dims));
    X = bsxfun(@minus, X, mean(X, 1)) * M;
    clear covX M lambda
    
    % Compute joint probabilities
    fprintf('Computing Probabilities\n');
    P = single(x2p(X, perplexity, 1e-5));                                   % compute affinities using fixed perplexity
	P = 0.5 * (P + P');                                                     % make symmetric
    P = P ./ sum(P(:));                                                     % obtain estimation of joint probabilities
    P = max(P, realmin('single'));
    P = P ./ sum(P(:));
    
    % Initialize the solution
    max_iter = 50;                                                          % maximum number of iterations
    base_eta = .6;                                                          % initial step size
    tol_step = 1e-5;                                                        % minimum allowable step size
    tol_improv = 1e-4;                                                      % minimum cost function improvement
    c1 = .02;                                                               % Armijo sufficient decrease parameter
    c2 = .9;                                                                % curvature parameter
    max_search = 10;                                                        % maximum number of function evaluations
    LS = 4;                                                                 % interpolation
    iter = 0; C = Inf; old_C = Inf; eta = Inf;
    const = sum(P(:) .* log(P(:)));
    Y = single(X(:,1:no_dims));
    
    % Perform the optimization
    fprintf('Performing optimization');
    Cs = nan(max_iter, 1); ts = nan(max_iter, 1); tic
    while iter < max_iter && eta > tol_step && (iter == 0 || (old_C - C) > tol_improv)
        
        fprintf('Iteration %d...\n', iter);
        % Compute joint probability that point i and j are neighbors
        iter = iter + 1; old_C = C;
        [C, dC, direction] = tsne_grad(Y(:), P, const);                     % we already have C and dC for iter > 1
        
        % Do a linesearch that satisfies the Wolfe conditions
        gtd = dC' * direction;                                              % directional derivative
        if gtd > -1e-9                                                      % sanity check
            disp(['Directional derivative became near zero or positive (value is ' num2str(gtd) ').']);
            Cs(iter) = C;
            ts(iter) = toc;
            break;
        end                                                     
        [eta, C, dC, no_eval] = WolfeLineSearch(Y(:), base_eta, direction, C, dC, gtd, c1, c2, LS, max_search, tol_step, false, false, false, @tsne_grad, P, const);
        disp(['Iteration ' num2str(iter) ': C = ' num2str(C) ' (eta = ' num2str(eta) ', ' num2str(no_eval) ' function evaluations)']);
        Y = Y + eta * reshape(direction, size(Y));
        
        % Display scatter plot (maximally first three dimensions)
        if ~rem(iter, 1) && ~isempty(labels)
            if no_dims == 1
                scatter(Y, Y, 12, labels, 'filled');
            elseif no_dims == 2
                scatter(Y(:,1), Y(:,2), 12, labels, 'filled');
            else
                scatter3(Y(:,1), Y(:,2), Y(:,3), 40, labels, 'filled');
            end
            axis tight
            drawnow
        end
        
        % Store error and time
        Cs(iter) = C;
        ts(iter) = toc;
    end
    Cs = Cs(1:iter);
    ts = ts(1:iter);
