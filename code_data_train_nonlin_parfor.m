Fs = 8000;
beta = 0.9;
Fc = 3600;

WavScp = args{1};
MfcScp = args{2};
WavList = textread(WavScp,'%s');
MfcList = textread(MfcScp,'%s');

WavScp_test = args{3};
MfcScp_test = args{4};
WavList_test = textread(WavScp_test,'%s');
MfcList_test = textread(MfcScp_test,'%s');

%{
WavList = {'/media/Data2/hitesh/databases/aurora4/train_clean/20h_8k/20hc0109.wv1', ...
    '/media/Data2/hitesh/databases/aurora4/train_clean/20h_8k/20hc010e.wv1', ...
    '/media/Data2/hitesh/databases/aurora4/train_clean/20h_8k/20ho0107.wv1', ...
    '/media/Data2/hitesh/databases/aurora4/train_clean/20h_8k/20ho010o.wv1', ...
    '/media/Data2/hitesh/databases/aurora4/train_clean/20h_8k/20ho0104.wv1', ...
    '/media/Data2/hitesh/databases/aurora4/train_clean/20h_8k/20ho0108.wv1'};
MfcList = {'1.MFC', '2.MFC', '3.MFC', '4.MFC', '5.MFC', '6.MFC'};
WavList_test = {'/media/Data2/hitesh/databases/aurora4/test_clean/444_8k/444c020p.wv1', ...
    '/media/Data2/hitesh/databases/aurora4/test_clean/444_8k/444c020j.wv1'};
MfcList_test = {'7.MFC', '8.MFC'};
%}

L_train = length(WavList);
L_test = length(WavList_test);

num_frames_train = zeros(L_train,1);
num_frames_test = zeros(L_test,1);

feat_cat = [];   

matlabpool Profile1 8;
data(length(L_train)).gabor = [];
data(length(L_train)).pncc = [];
data1(length(L_test)).gabor = [];
data1(length(L_test)).pncc = [];

parfor k = 1:L_train
    
    srcfile = WavList{k};
    
    fid2 = fopen(srcfile,'r');
    rawdata = fread(fid2,'int16','b');
    fclose(fid2);
    
    rawdata = single(rawdata/(1.25*max(rawdata)));
    gabor = single(gbfb_fe(rawdata, Fs));
    pncc = single(PNCC(rawdata, Fs));
    
    num_frames_train(k) = size(gabor,1);
    data(k).gabor = gabor;
    data(k).pncc = pncc;

   fprintf('file %s done.\n',srcfile);
end


%% includes test part also for manifold mapping



parfor k = 1:L_test
    
    srcfile = WavList_test{k};
    
    fid2 = fopen(srcfile,'r');
    rawdata = fread(fid2,'int16','b');
    fclose(fid2);
    
    rawdata = single(rawdata/(1.25*max(rawdata)));
    gabor = single(gbfb_fe(rawdata, Fs));
    pncc = single(PNCC(rawdata, Fs));
    
    num_frames_test(k) = size(gabor,1);
    data1(k).gabor =gabor;
    data1(k).pncc = pncc;
    
    fprintf('file %s done.\n',srcfile);
end

matlabpool close;
outputsize = 13;

fprintf('Performing Laplacian EigenMap reduction on Gabor features to reduce to %d dimension vector', outputsize);

num_neigh = 10;

[features] = leigs( single(vertcat(data(:).gabor,data1(:).gabor)), 'nn', num_neigh, outputsize);
disp('Laplace EigenMap Dimension Reduction done...');
disp('Writing feature files');

features = [ vertcat(data(:).pncc, data1(:).pncc) features];
clear data;
features = features./repmat(std(features), size(features,1), 1);


end_frame = 0;

for k = 1:L_train
    tgtfile = MfcList{k};
    
    start_frame = 1 + end_frame;
    end_frame = start_frame + num_frames_train(k) - 1;
    writehtk(tgtfile,features(start_frame:end_frame,:), 0.01,9);
end


for k = 1:L_test
    tgtfile = MfcList_test{k};
    
    start_frame = 1 + end_frame;
    end_frame = start_frame + num_frames_test(k) - 1;
    writehtk(tgtfile,features(start_frame:end_frame,:), 0.01,9);
end

clear features num_frames_train num_frames_test;

% fclose(fid);
