function writehtk(file,d,fp,tc)
%WRITEHTK write data in HTK format []=(FILE,D,FP,TC)
% d is data array, fp is frame period in seconds, tc is typecode
% tc is the sum of the following values:
%			0		WAVEFORM
%			1		LPC
%			2		LPREFC
%			3		LPCEPSTRA
%			4		LPDELCEP
%			5		IREFC
%			6		MFCC
%			7		FBANK
%			8		MELSPEC
%			9		USER
%			10		DISCRETE
%			64		-E		Includes energy terms
%			128	_N		Suppress absolute energy
%			256	_D		Include delta coefs
%			512	_A		Include acceleration coefs
%			1024	_C		Compressed (not implemented yet)
%			2048	_Z		Zero mean static coefs
%			4096	_K		CRC checksum (not implemented yet)
%			8192	_0		Include 0'th cepstral coef

%      Copyright (C) Mike Brookes 1997
%
%      Last modified Tue May 12 16:11:32 1998
%
%   VOICEBOX home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   This program is free software; you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation; either version 2 of the License, or
%   (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You can obtain a copy of the GNU General Public License from
%   ftp://prep.ai.mit.edu/pub/gnu/COPYING-2.0 or by writing to
%   Free Software Foundation, Inc.,675 Mass Ave, Cambridge, MA 02139, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fid=fopen(file,'wb');
if fid < 0
   error(sprintf('Cannot write to file %s',file));
end

[nf,nv]=size(d);
hb=floor(tc*pow2(-14:-6));
hd=hb(9:-1:2)-2*hb(8:-1:1);
dt=tc-64*hb(9);
if hd(7)>0
   error('CRC check not implemented in this version');
end
if hd(5)>0
   error('compression not implemented in this version');
end

fwrite(fid,nf,'int32');
fwrite(fid,round(fp*1.E7),'int32');
fwrite(fid,nv*4,'int16');
fwrite(fid,tc,'int16');

fwrite(fid,d.','float');

fclose(fid);