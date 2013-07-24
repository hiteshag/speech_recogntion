function [features, gfilters] = gbfb_fe(signal, fs)
%usage: [features, gfilters] = gbfb_fe(signal, fs)
%   signal:   waveform signal
%   fs:       sampling rate in Hz
% 
%   features: Gabor filter bank (GBFB) features 
%   gfilters: Gabor filter bank filters
%
% - Gabor Filter Bank Feature Extraction v2.0 -
%
% Autor    : Marc René Schädler
% Email    : marc.r.schaedler@uni-oldenburg.de
% Institute: Medical Physics / Carl-von-Ossietzky University Oldenburg, Germany
% 
%-----------------------------------------------------------------------------
%
% Licensing of the feature extraction code: Dual-License
% 
% The Gabor Filterbank (GBFB) feature extraction code is licensed under both 
% General Public License (GPL) version 3 and a proprietary license that can be 
% arranged with us. In practical sense, this means:
% 
% - If you are developing Open Source Software (OSS) based on the GBFB code, 
%   chances are you will be able to use it freely under GPL. But please double check 
%   http://www.gnu.org/licenses/license-list.html for OSS license compatibility with GPL
% - Alternatively, if you are unable to release your application as Open Source Software, 
%   you may arrange alternative licensing with us. Just send your inquiry to the 
%   author to discuss this option.
% 
%-----------------------------------------------------------------------------
% 
% Release Notes:
% This script is written to produce the results published in [1]
% For an overview of current publications / further developments of this 
% frontend, visit http://medi.uni-oldenburg.de/GBFB
% 
% [1] M.R. Schädler, B.T. Meyer, B. Kollmeier
% "Spectro-temporal modulation subspace-spanning filter bank features 
% for robust automatic speech recognition ", J. Acoust. Soc. Am. Volume 131, 
% Issue 5, pp. 4134-4151 (2012)
%
% Paper URL: http://link.aip.org/link/?JAS/131/4134
% Paper DOI: 10.1121/1.3699200
%

%% Default settings (as used in [1])

% addpath FastICA_25/

% Mel-spectrogram settings
win_shift   = 10;           % ms
win_length  = 25;           % ms
freq_range  = [  64 4000]; 	% Hz

%num_bands   = 23;           % For Mel
 num_bands   = 40;           % For Power normalized Mel

skip_frames = [   2    2];  %


% Filter bank settings [spectral temporal]
omega_max   = [pi/2 pi/2]; 	% radian
size_max    = [3*23   40]; 	% bands, frames
nu          = [ 3.5  3.5]; 	% half-waves under envelope
distance    = [ 0.3  0.2]; 	% controls the spacing of filters
% Changed spectral distance to 0.2 from 0.3


%% Set up Gabor filter bank with given parameters

% Calculate center modulation frequencies.
[omega_n, omega_k] = gfbank_calcaxis(omega_max, size_max, nu, distance);
omega_n_num = length(omega_n);
omega_k_num = length(omega_k);

% Generate filters for all pairs of spectral and temporal modulation
% frequencies except for the redundant ones.
gfilters = cell(omega_k_num,omega_n_num);
for i=1:omega_k_num
    for j=1:omega_n_num
        if ~(omega_k(i) < 0 && omega_n(j) == 0)
            gfilters{i,j} = gfilter_gen(omega_k(i), omega_n(j), nu(1), nu(2), size_max(1), size_max(2));
        end
    end
end


%% Calculate mel spectrogram.
%mel_spec = mel_spectrogram(signal, fs, win_shift, win_length, freq_range, num_bands);

mel_spec = PNCC_spectrogram(signal, fs, win_shift, win_length, freq_range, num_bands);

% Skip frames that might contain padded zeros, as the first frame is centered 
% on the first sample and the last frame might be centered on the last sample
% mel_spec = skip(mel_spec, skip_frames);

% Logarithmic compression.
%mel_spec_log = log(max(exp(-50), mel_spec));

mel_spec_log = mel_spec;


%% Filter mel spectrogram with filter bank filters and select representative channels.
gfilters_output = cell(omega_k_num,omega_n_num);
for i=1:omega_k_num
    for j=1:omega_n_num
        gfilter = gfilters{i,j};
        if ~isempty(gfilter)
            % Filter mel spectrogram with Gabor filter.
            mel_spec_log_filtered = gfilter_filter(gfilter, mel_spec_log);
            % Select representative channels from filtered Mel-spectrogram.
            gfilters_output{i,j} = gfilter_rep(gfilter, mel_spec_log_filtered);
        end
    end
end
features = cell2mat(reshape(gfilters_output,[],1));

% Use the real part of the filter output
features = real(features);

% Mean and variance normalization was not used in the paper but results in 
% important improvements. To activate it uncomment the following lines.
features_mean = mean(features,2);
features_std = sqrt(var(features,1,2));
features = (features - repmat(features_mean,1,size(features,2))) ./ repmat(features_std,1,size(features,2));

features = features';	% To get (no.of frames) x (dim. of feature vector)

% features = fastica(features, 'stabilization', 'on', 'numofic', 39, 'verbose', 'off');        
% features = features';

% outputsize = 39;
% [~, score] = princomp(features);
% features = score(:,1:outputsize);

end


function [omega_n, omega_k] = gfbank_calcaxis(omega_max, size_max, nu, distance)
% Calculates the modulation center frequencies iteratively.
% Termination condition for iteration is reaching omega_min, which is
% derived from size_max.
omega_min = (pi .* nu) ./ size_max;

% Eq. (2b)
c = distance .* 8 ./ nu;
% Second factor of Eq. (2a)
space_n = (1 + c(2)./2) ./ (1 - c(2)./2);
count_n = 0;
omega_n(1) = omega_max(2);
% Iterate starting with omega_max in spectral dimension
while omega_n/space_n > omega_min(2)
    omega_n(1+count_n) = omega_max(2)/space_n.^count_n;
    count_n = count_n + 1;
end
omega_n = fliplr(omega_n);
% Add DC
omega_n = [0,omega_n];
% Second factor of Eq. (2a)
space_k = (1 + c(1)./2) ./ (1 - c(1)./2);
count_k = 0;
omega_k(1) = omega_max(1);
% Iterate starting with omega_max in temporal dimension
while omega_k/space_k > omega_min(1)
    omega_k(1+count_k) = omega_max(1)/space_k.^count_k;
    count_k = count_k + 1;
end
% Add DC and negative MFs for spectro-temporal opposite 
% filters (upward/downward)
omega_k = [-omega_k,0,fliplr(omega_k)];
end


function gfilter = gfilter_gen(omega_k, omega_n, nu_k, nu_n, size_max_k, size_max_n)
% Generates a gabor filter function with:
%  omega_k       spectral mod. freq. in rad
%  omega_n       temporal mod. freq. in rad
%  nu_k          number of half waves unter the envelope in spectral dim.
%  nu_n          number of half waves unter the envelope in temporal dim.
%  size_max_k    max. allowed extension in spectral dimension
%  size_max_n    max. allowed extension in temporal dimension

% Calculate windows width.
w_n = 2*pi / abs(omega_n) * nu_n / 2;
w_k = 2*pi / abs(omega_k) * nu_k / 2;

% If the size exceeds the max. allowed extension in a dimension set the
% corresponding mod. freq. to zero.
if w_n > size_max_n
    w_n = size_max_n;
    omega_n = 0;
end
if w_k > size_max_k
    w_k = size_max_k;
    omega_k = 0;
end

% Separable hanning envelope, cf. Eq. (1c).
env_n = hann_win(w_n-1);
env_k = hann_win(w_k-1);
envelope = env_k * env_n.';
[win_size_k, win_size_n] = size(envelope);

% Sinusoid carrier, cf. Eq. (1c).
n_0 = (win_size_n+1) / 2;
k_0 = (win_size_k+1) / 2;
[n,k] = meshgrid (1:win_size_n, 1:win_size_k);
sinusoid = exp(1i*omega_n*(n - n_0) + 1i*omega_k*(k - k_0));

% Eq. 1c
gfilter  = envelope .* sinusoid;

% Compensate the DC part by subtracting an appropiate part
% of the envelope if filter is not the DC filter.
envelope_mean = mean(mean(envelope));
gfilter_mean = mean(mean(gfilter));
if (omega_n ~= 0) || (omega_k ~= 0)
    gfilter = gfilter - envelope./envelope_mean .* gfilter_mean;
else
    % Add an imaginary part to DC filter for a fair real/imag comparison.
    gfilter = gfilter + 1i*gfilter;
end
% Normalize filter to have gains <= 1.
gfilter = gfilter ./ max(max(abs(fft2(gfilter))));
end


function log_mel_spec_filt = gfilter_filter(gfilter, log_mel_spec)
% Applies the filtering with a 2D Gabor filter to log_mel_spec
% This includes the special treatment of filters that do not lie fully
% inside the spectrogram
if any(any(gfilter < 0))
    % Compare this code to the compensation for the DC part in the
    % 'gfilter_gen' function. This is an online version of it removing the
    % DC part of the filters by subtracting an appropriate part of the
    % filters' envelope.
    gfilter_abs_norm = abs(gfilter) ./ sum(sum(abs(gfilter)));
    gfilter_dc_map = fftconv2(ones(size(log_mel_spec)), gfilter,'same');
    env_dc_map = fftconv2(ones(size(log_mel_spec)), gfilter_abs_norm,'same');
    dc_map = fftconv2(log_mel_spec, gfilter_abs_norm,'same') ./ env_dc_map .* gfilter_dc_map;
else
    % Dont' remove the DC part if it is the DC filter.
    dc_map = 0;
end
% Filter log_mel_spec with the 2d Gabor filter and remove the DC parts.
log_mel_spec_filt = fftconv2(log_mel_spec, gfilter,'same') - dc_map;
end


function mel_spec_rep = gfilter_rep(gfilter, mel_spec)
% Selects the center channel by choosing k_offset and those with k_factor
% channels distance to it in spectral dimension where k_factor is approx.
% 1/4 of the filters extension in the spectral dimension.
k_factor = floor(1/4 * size(gfilter,1));
if k_factor < 1
    k_factor = 1;
end
k_offset = mod(floor(size(mel_spec,1)/2),k_factor);
k_idx = (1+k_offset):k_factor:size(mel_spec,1);
% Apply subsampling.
mel_spec_rep = mel_spec(k_idx,:);
end


function window_function = hann_win(width)
% A hanning window function that accepts non-integer width and always
% returns a symmetric window with an odd number of samples.
x_center = 0.5;
x_values = [fliplr((x_center-1/(width+1)):-1/(width+1):0), x_center:1/(width+1):1]';
valid_values_mask = (x_values > 0) & (x_values < 1);
window_function =  0.5*(1-(cos(2*pi*x_values(valid_values_mask))));
end

function out = fftconv2(in1, in2, shape)
% 2D convolution in terms of the 2D FFT that substitutes conv2(in1, in2, shape).
size_y = size(in1,1)+size(in2,1)-1;
size_x = size(in1,2)+size(in2,2)-1;
fft_size_x = 2.^ceil(log2(size_x));
fft_size_y = 2.^ceil(log2(size_y));
in1_fft = fft2(in1,fft_size_y,fft_size_x);
in2_fft = fft2(in2,fft_size_y,fft_size_x);
out_fft = in1_fft .* in2_fft;
out_padd = ifft2(out_fft);
out_padd = out_padd(1:size_y,1:size_x);
switch shape
    case 'same'
        y_offset = floor(size(in2,1)/2);
        x_offset = floor(size(in2,2)/2);
        out = out_padd(1+y_offset:size(in1,1)+y_offset,1+x_offset:size(in1,2)+x_offset);
    case 'full'
        out = out_padd;
end
end

function mel_spec = mel_spectrogram(signal, fs, win_shift, win_length, freq_range, num_bands)
% Calculates a Mel-spectrogram of a signal with sampling rate fs.
if size(signal,2) == 1
    signal = signal';
end
N = round(win_length/1000*fs);
M = round(win_shift/1000*fs);
num_coeff = 2^(ceil(log2(N)));
signal = filter([1,-1],[1,-0.999],signal);
signal = (signal - mean(signal)) ./ sqrt(var(signal,0));
signal = [zeros(1, floor(N/2)-1), signal, zeros(1, floor(N/2)+2)];
num_frames = 1 + floor ((length(signal) - N) / M);
signal_frame = zeros(N,num_frames);
for i = 1:num_frames
    signal_frame(:,i) = signal(1+(i-1)*M:N+(i-1)*M);
end
window_frame = hamming(N) * ones(1,num_frames);
signal_frame = signal_frame .* window_frame;
spec = abs(fft(signal_frame,num_coeff,1));
freq_centers = mel2hz(linspace(hz2mel(freq_range(1)),hz2mel(freq_range(2)),num_bands+2));
mel_spec = triafbmat(freq_centers, num_coeff, fs) * spec;
end


function transmat = triafbmat(freq_centers, num_coeff, fs)
% Generates a triangular filterbank transformation matrix for the given
% center frequencies.
freq_centers_idx = round(freq_centers/fs * num_coeff);
num_bands = length(freq_centers)-(1+1);
transmat = zeros(num_bands, num_coeff);
for i=1:num_bands
    left = freq_centers_idx(i);
    center = freq_centers_idx(i+1);
    right = freq_centers_idx(i+1+1);
    if (left >= 1)
        transmat(i,left:center) = linspace(0, 1, center-left+1);
    end
    if (right <= num_coeff)
        transmat(i,center:right) = linspace(1, 0, right-center+1);
    end
end
end


%GAMMATONE Create a gammatone filter bank.
% [H,FC,T,F] = GAMMATONE(NUM,LEN,FMIN,FMAX,FS) creates
% a gammatone filter bank containing NUM filters of length LEN
% samples. The center frequency of the first filter is FMIN Hz
% and the maximum frequency of the filter bank is FMAX Hz.
% The generated filter bank is returned as a matrix of size NUM x
% LEN transfer functions, a vector FC of center frequencies, and
% the times T and frequencies F over which the filters are
% constructed.
%
% [H,FC,T,F] = GAMMATONE(NUM,LEN,FMIN,FMAX,FS,PAD_BW) pads the
% highest band in the filter bank is by PAD_BW to avoid aliasing; by
% default, PAD_BW == 2; if PAD_BW == 1, then FC(end) == FMAX.

% Authors: Eftychios A. Pnevmatikakis and Robert J. Turetsky
% Copyright 2009-2012 Eftychios A. Pnevmatikakis and Robert J. Turetsky

function [h,fc,t,f] = gammatone(num,len,fmin,fmax,fs,pad_bw)

if exist('pad_bw') ~= 1
    pad_bw = 2;
end

EarQ = 9.26449;
minBW = 24.7;
order = 4;
dt = 1/fs;
t = dt*(0:len-1);
f = (0:length(t)-1)/length(t)*fs;
beta = 1.019;

Wp = fmax;
fmax = EarQ*(Wp-pad_bw*beta*minBW)/(EarQ+pad_bw*beta);

overlap = EarQ*(log(fmax+EarQ*minBW)-log(fmin+EarQ*minBW))/max(1,num-1);
fc = -EarQ*minBW + (fmax+EarQ*minBW)*exp(-(num-(1:num))*overlap/EarQ);
h = zeros(num,len);
for i=1:num,
    h(i,:) = t.^(order-1).*exp(-2*pi*beta*(fc(i)/EarQ+minBW)*t).*cos(2*pi*fc(i)*t);
    h(i,:) = h(i,:)/max(abs(fft(h(i,:))));
end
end


function output = skip(input, values)
% Skips values(1) frames at the beginning
% and values(2) frames at the end if possible.
start = 1+values(1);
stop = size(input,2)-values(2);
while  stop - start <= 0
    stop    = min(size(input,2),stop+1);
    start   = max(1,start-1);
end
output = input(:,start:stop);
end


function f = mel2hz (m)
% Converts frequency from Mel to Hz
f = 700.*((10.^(m./2595))-1);
end


function m = hz2mel (f)
% Converts frequency from Hz to Mel
m = 2595.*log10(1+f./700);
end
