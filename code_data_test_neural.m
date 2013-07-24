Fs = 8000;
beta = 0.9;
Fc = 3600;

WavScp = args{1};
MfcScp = args{2};
WavList = textread(WavScp,'%s');
MfcList = textread(MfcScp,'%s');

% WavList = {'/media/Data2/hitesh/databases/aurora4/train_clean/20h_8k/20hc0109.wv1', ...
%    '/media/Data2/hitesh/databases/aurora4/train_clean/20h_8k/20hc010e.wv1', ...
%    '/media/Data2/hitesh/databases/aurora4/train_clean/20h_8k/20ho0107.wv1', ...
%    '/media/Data2/hitesh/databases/aurora4/train_clean/20h_8k/20ho010o.wv1', ...
%    '/media/Data2/hitesh/databases/aurora4/train_clean/20h_8k/20ho0104.wv1', ...
%    '/media/Data2/hitesh/databases/aurora4/train_clean/20h_8k/20ho0108.wv1'};
% MfcList = {'1.MFC', '2.MFC', '3.MFC', '4.MFC', '5.MFC', '6.MFC'};

% load('neural_net_1000_20.mat');
load('neural_net_1000_pca.mat');
load('coeff_gabor.mat');

for k = 1:length(WavList)
    srcfile = WavList{k};
    tgtfile = MfcList{k};
    
    fid2 = fopen(srcfile,'r');
    rawdata = fread(fid2,'int16','b');
    fclose(fid2);
    
    rawdata = single(rawdata/(1.25*max(rawdata)));
    gabor = single(gbfb_fe(rawdata, Fs));
    pncc = single(PNCC(rawdata, Fs));
    
%     gabor_leigs = net(gabor');
    gabor_leigs = sim(net, gabor');
    feat_norm = [pncc gabor_leigs'];
%     gabor_leigs = gabor * coeff_gabor;
%     feat_norm = [pncc gabor_leigs];
%     feat_norm = pncc;
    
   feat_neural = feat_norm./repmat(std(feat_norm),size(feat_norm,1),1);

   writehtk(tgtfile,feat_neural,0.01,9);
   fprintf('file %s done.\n',srcfile);
   
   if(k == round(0.25*length(WavList)))
       disp('features extracted for 25% of training data');
   elseif(k == round(0.5*length(WavList)))
       disp('features extracted for 50% of training data');
   elseif(k == round(0.75*length(WavList)))
       disp('features extracted for 75% of training data');
   elseif(k == length(WavList))
       disp('features extracted for 100% of training data');
   end    
end

