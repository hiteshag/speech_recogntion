Fs = 8000;
beta = 0.9;
Fc = 3600;

WavScp = args{1};
MfcScp = args{2};
WavList = textread(WavScp,'%s');
MfcList = textread(MfcScp,'%s');

load('neural_net.mat');

matlabpool Profile1 8;

parfor k = 1:length(WavList)
    srcfile = WavList{k};
    tgtfile = MfcList{k};
    
    fid2 = fopen(srcfile,'r');
    rawdata = fread(fid2,'int16','b');
    fclose(fid2);
    
    rawdata = single(rawdata/(1.25*max(rawdata)));

    gabor = single(gbfb_fe(rawdata, Fs));
    pncc = PNCC(rawdata, Fs);

    gabor_leigs = (net(gabor'))';
    feat_norm = [single(pncc) gabor_leigs];
    
    feat_leigs = feat_norm./repmat(std(feat_norm),size(feat_norm,1),1);

    writehtk(tgtfile,feat_leigs,0.01,9);
    fprintf('file %s done.\n',srcfile);

%     if(k == round(0.25*length(WavList)))
%         disp('features extracted for 25% of training data');
%     elseif(k == round(0.5*length(WavList)))
%         disp('features extracted for 50% of training data');
%     elseif(k == round(0.75*length(WavList)))
%         disp('features extracted for 75% of training data');
%     elseif(k == length(WavList))
%         disp('features extracted for 100% of training data');
%     end    

end

matlabpool close;
