Fs = 8000;
beta = 0.9;
Fc = 3600;

WavScp = args{1};
MfcScp = args{2};
WavList = textread(WavScp,'%s');
MfcList = textread(MfcScp,'%s');

dim_red_type = args{3};

flag = 2;	% Flag to skip data extraction and kmeans. Skip data extraction and kmeans if not 0, dimension reduction if not 1.

% WavList = {'/media/Data2/hitesh/databases/aurora4/train_clean/20h_8k/20hc0109.wv1', ...
%     '/media/Data2/hitesh/databases/aurora4/train_clean/20h_8k/20hc010e.wv1', ...
%     '/media/Data2/hitesh/databases/aurora4/train_clean/20h_8k/20ho0107.wv1', ...
%     '/media/Data2/hitesh/databases/aurora4/train_clean/20h_8k/20ho010o.wv1', ...
%     '/media/Data2/hitesh/databases/aurora4/train_clean/20h_8k/20ho0104.wv1', ...
%     '/media/Data2/hitesh/databases/aurora4/train_clean/20h_8k/20ho0108.wv1'};
% MfcList = {'1.MFC', '2.MFC', '3.MFC', '4.MFC', '5.MFC', '6.MFC'};

%% Feature Extraction - gabor and pncc concatenated features

if flag == 0
    L_train = length(WavList);
    num_frames_train = zeros(L_train,1);
    
    data(length(L_train)).gabor = [];
    % data(length(L_train)).pncc = [];
    data(length(L_train)).mfcc = [];
    
    matlabpool Profile1 8;
    parfor k = 1:L_train
        srcfile = WavList{k};
        fid2 = fopen(srcfile,'r');
        rawdata = fread(fid2,'int16','b');
        fclose(fid2);
        
        rawdata = single(rawdata/(1.25*max(rawdata)));
        gabor = single(gbfb_fe(rawdata, Fs));
        pncc = single(PNCC(rawdata, Fs));
        mfcc = single(MFCC(rawdata, Fs));
        
        num_frames_train(k) = size(gabor,1);
        data(k).gabor = gabor;
        %     data(k).pncc = pncc;
        data(k).mfcc = mfcc;
        
        fprintf('file %s done.\n',srcfile);
    end
    matlabpool close;
    %% Clustering to find most representative frames (tentative)
    matlabpool Profile1 8;
    kmeans_size = 1000;
    mfcc_cat = vertcat(data(:).mfcc);
    fprintf('Performing k-means to find most represntative frames\n');
    options = statset('Display', 'iter', 'UseParallel', 'always', 'MaxIter', 150);
    [idx, ~] = kmeans(mfcc_cat, kmeans_size, 'distance', 'cityblock', 'onlinephase', 'off',  'Options', options);
    save('cluster_ids.mat', 'idx');
    % addpath('kmeans_algos/vggkmeans/');
    % CX = Kmeans((vertcat(data(:).gabor))', kmeans_size);
    % clear data;
    matlabpool close;
    
    %% Sub-sampling of data to get 5% points from each cluster
    fprintf('Sub-sampling data\n');
    gabor_cat = vertcat(data(:).gabor);
    gabor_kmeans = [];
    for i = 1:kmeans_size
        indices = find(idx==i);
        len = ceil(0.07*length(indices));
        index = randperm(length(indices));
        gabor_kmeans = [gabor_kmeans; gabor_cat(indices(index(1:len)),:)];
    end
    save('kmeans_data.mat', 'gabor_kmeans');
    clear gabor_cat idx data;
end

%% Non-linear dimension reduction
filename = sprintf('dim_red_data_%s.mat', dim_red_type);
if flag == 1
load('kmeans_data.mat');
% Laplace EigenMaps
if strcmp(dim_red_type, 'leigs')
    outputsize = 14;
    fprintf('Performing Laplacian EigenMap reduction on Gabor features to reduce to %d dimension vector\n', outputsize);
    num_neigh = 10;
    % % features = leigs(vertcat(data(:).gabor), 'nn', num_neigh, outputsize);
    features = leigs(gabor_kmeans, 'nn', num_neigh, outputsize);
    features = features(:,1:outputsize-1);
    disp('Laplace EigenMap Dimension Reduction done...');
end

% t-SNE
if strcmp(dim_red_type, 't-sne')
    outputsize = 13;
    addpath('nonlinDR/SNE/tSNE_new_optimizer/');
    fprintf('Performing t-SNE on Gabor features to reduce to %d dimension vector\n', outputsize);
    % features = fast_tsne(vertcat(data(:).gabor), [], outputsize);   % Optimized t-SNE
    features = fast_tsne(gabor_kmeans, [], outputsize);   % Optimized t-SNE
    disp('t-SNE done...');
end

% PCA
if strcmp(dim_red_type, 'pca')
    outputsize = 13;
    fprintf('Performing PCA on Gabor features to reduce to %d dimension vector\n', outputsize);
    [Coeff_gabor, features] = princomp(gabor_kmeans, 'econ');
    coeff_gabor = Coeff_gabor(:, 1:outputsize);
    save('coeff_gabor.mat', 'coeff_gabor');
    features = features(:, 1:outputsize);
    disp('PCA done...');
end
save(filename, 'features');
end

load('kmeans_data.mat');
load(filename);
%% MVN of features before training neural network
for i = 1:size(features,2)
        features(:,i) = (features(:,i) - mean(features(:,i)))/std(features(:,i));
end

%% Neural Network training to determine mapping of features

disp('Training neural network now');
net = feedforwardnet;
net.efficiency.memoryReduction = 3;
net.trainParam.showWindow = false;
net.trainParam.showCommandLine = false;
% net.trainParam.show = 1;
net.trainParam.epochs = 77;
net.trainParam.time = 43200;    % Maximum time in seconds (12 hours)
net.trainFcn = 'trainbr';   % For Bayesian regularization - trainbr, Small memory requirements - trainscg, trainrp
% [net,~] = train(net, (vertcat(data(:).gabor))', features');
[net,~] = train(net, gabor_kmeans', features');
% nntraintool('close');
filename = sprintf('neural_net_%s.mat', dim_red_type);
save(filename, 'net');
clear net;
clear gabor_kmeans features;

% features = [ vertcat(data(:).pncc) features];
% features = features./repmat(std(features), size(features,1), 1);
% clear data;

%% Writing feature files
% disp('Writing feature files');
% end_frame = 0;
%
% for k = 1:L_train
%     tgtfile = MfcList{k};
%
%     start_frame = 1 + end_frame;
%     end_frame = start_frame + num_frames_train(k) - 1;
%     writehtk(tgtfile,features(start_frame:end_frame,:), 0.01,9);
% end

clear features num_frames_train;

% fclose(fid);
