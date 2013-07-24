Fs = 8000;
beta = 0.9;
Fc = 3600;

WavScp = args{1};
MfcScp = args{2};


WavList = textread(WavScp,'%s');
MfcList = textread(MfcScp,'%s');

%WavList = {'/home/anirudh/Dropbox/acads/Speech_214B/EE214B_project/databases/aurora4/male_train/024o030n.wv1'};
%MfcList = {''};

num_frames = zeros(length(WavList),1);

feat_cat = [];    % Change 311 if size of Gabor feature vector changes
				   % Row size = 8000*500=4000000
addpath('FastICA_25/');

end_frame = 0;

for k = 1:length(WavList)
    
    srcfile = WavList{k};
    tgtfile = MfcList{k};
    
    fid2 = fopen(srcfile,'r');
    rawdata = fread(fid2,'int16','b');
    fclose(fid2);
    
    rawdata = rawdata/(1.25*max(rawdata));

    gabor = gbfb_fe(rawdata, Fs);
    
    num_frames(k) = size(gabor,1);
    start_frame = 1 + end_frame;
    end_frame = start_frame + num_frames(k) - 1;
    feat_cat(start_frame:end_frame, :) = gabor;
    
%     writehtk(tgtfile,gabor,0.01,9);
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

outputsize = 13;

disp('Performing PCA on Gabor features to reduce to 39 dimension vector');

[Coeff, score, EigenVal] = princomp(feat_cat);
features = score(:,1:outputsize);

save('EigenVal.mat', 'EigenVal');
save('Coeff.mat', 'Coeff');
disp('PCA done...');

% disp('Performing ICA on Gabor features');
% [score, mixing, separating] = fastica(feat_cat', 'stabilization', 'on', 'numofic', outputsize, 'verbose', 'off');
% features = score';
% save('mixing.mat', 'mixing');
% save('separating.mat', 'separating');
% disp('ICA done');

disp('Writing feature files');

end_frame = 0;

for k = 1:length(WavList)
    % srcfile = WavList{k};
    tgtfile = MfcList{k};
    
    start_frame = 1 + end_frame;
    end_frame = start_frame + num_frames(k) - 1;
    writehtk(tgtfile,features(start_frame:end_frame,:), 0.01,9);
end

clear feat_cat features score num_frames;

% fclose(fid);
