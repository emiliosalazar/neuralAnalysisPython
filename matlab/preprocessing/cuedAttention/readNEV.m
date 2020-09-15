function [spike] = readNEV(nevfile)
%function [spike] = readNEV(nevfile)
%
% readNEV takes an NEV file as input and returns the event codes
% and times. 
%
% The columns of "spike" are channel, spike class (or digital port
% value for digital events), and time (seconds). The channel is 0
% for a digital event, and 1 and above for spike channels.
%
% This is a mex-optimized version of the matlab NEV_reader.m file
%
% NOTE (9/19/2012 MAS): This file now works on 64-bit Mac and
% Windows. On 64-bit Linux it should work but isn't thoroughly
% tested. Don't run it on a 32-bit system! Instead use the
% readNEV_32bit file.
%
