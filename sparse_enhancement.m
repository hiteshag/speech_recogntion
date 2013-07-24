[rawdata , fs]= wavread('male.wav');
frames = get_frames(rawdata,fs);


for i=1:size(frames,2)
    frm_cur = frames(:,i);
    
    frm_spec = fft(frm_cur);
    
    
    