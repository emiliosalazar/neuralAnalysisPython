function [slabel,spikes, net_labels] = runNASNet(filename,gamma,net_name,varargin)
%
% This script classifies waveforms using a trained neural network
% (see Issar et al (2020)). The file can either be an NEV or a .mat file 
% containing a variable named 'waveforms'
%
%INPUTS:
%       filename- string filename of a .nev file OR a .mat file.
%                 IF it's a .mat file, MUST include a variable called
%                 "waveforms", waveforms is an Nx52 array, where N is the
%                 number of waveforms, and 52 is the number of samples in
%                 each waveform* (see NOTES below)
%       gamma- minimum P(spike) value for a waveform to be classified as a
%              spike (between 0 and 1). If you want a more lenient sort 
%              (i.e. allows for more noise), then choose a smaller gamma.
%       net_name- string of network name. All four NASNet output
%                files must be available and saved as the network name
%                followed by _w_hidden, _w_output, _b_hidden, _b_output.
%
%OUTPUTS:
%       slabel- list of labels for each waveform (0 for noise, 1 for spike)
%       spikes- waveform information (channel, sortcode, time)
%
% OPTIONAL ARGUMENTS:
% 'channels' - read all is default, or if a number list is specified only
%              those will be read
% 'writelabels' - false is default. If true, the classification labels will
%                 be written as sort codes into the nev file

%NOTES:
%****The number of samples in each waveform must match the number of
%    samples in the waveforms the network was trained on. The Issar et al
%    (2020) network was trained on waveforms with 52 samples.
%
% 02/2020: -Added channel selection functionality, as in read_nev
%          -Added clause to forbid overwritting digital codes (channel 0)
%           and uStim events (channels>512) to nev files.
%          -Writing into the nev file is optional now.

%%
% 512 is the highest available number of spike channels 
% (see Trellis NEV Spec manual)
maxspikech = 512;

% optional input arguments
p = inputParser;
p.addOptional('channels',[],@isnumeric);
p.addOptional('writelabels',false,@islogical);
p.parse(varargin{:});

ch          = p.Results.channels;
writelabels = p.Results.writelabels;

%% load trained network

try
    w1 = load(strcat(net_name,'_w_hidden'));
    b1 = load(strcat(net_name,'_b_hidden'));
    w2 = load(strcat(net_name,'_w_output'));
    b2 = load(strcat(net_name,'_b_output'));
catch
    error('The network you named does not exist or the files were not named appropriately.')
end

%% load file to classify

[~,~,ext] = fileparts(filename);
spikes = [];

wvLen = 30;

switch ext
    case '.nev'
        [spikes,waves] = read_nev(filename,'channels',ch);
        
        if any(spikes(:,1)==0)
            %These are digital codes. There are no waveforms to classify for
            %these indices so create placeholder in waveform list so indexing works
            %for moveToSortCode.
            waves(spikes(:,1)==0) = {ones(wvLen,1,'int16')};
        end
        
        waveforms = [waves{:}]'; %convert from a cell to an array
        clear waves;
        
    case '.mat'
        disp('loading data...')
        flVars = load(filename);
        fldnm = fieldnames(flVars);
        if length(fldnm)>1
            error(sprintf('input mat file must only have a single variable--the desired Nx%d matrix',wvLen))
        else
            waveforms = flVars.(fldnm{1});
        end
%        if ~exist('waveData'), error('waveforms must be stored in a variable named "waveData".'); end
        
    otherwise
        error(sprintf('file must be a .nev or Nx%d .mat', wvLen));
end

%check that the waveform size is appropriate for the network
if size(waveforms,2)~=size(w1,1)
    % erm... if it happens to have been saved with the labels in the first
    % column heh
    if size(waveforms, 2) == size(w1, 1)+1 && length(unique(waveforms(:, 1)))<=2
        waveforms = waveforms(:, 2:end);
    else
        error(['This network can only classify waveforms that are ' num2str(size(w1,1)),' samples long']);
    end
end

%% classify waveforms
nwaves = length(waveforms);
n_per_it = 1000; %number of waves per iteration
counter = 0;
nend = 0;

net_labels = zeros(1,nwaves);

disp('applying neural net classifier...')
while nend~=nwaves
    nstart = 1 + counter*n_per_it;
    nend = nstart + n_per_it - 1;
    
    if nend>nwaves %at the end of the file
        nend = nwaves;
    end
    
    n_loop = length(nstart:nend);
    
    % generate classifications
    wave_in = double(waveforms(nstart:nend,:));
    
    %*****Layer 1******
    layer1_raw = wave_in*w1 + repmat(b1',n_loop,1);
    
    %***ReLU activation****
    %layer1_out = log(1+exp(layer1_raw)); %faster ReLU
    layer1_out = max(0,layer1_raw); %more accurate ReLU
    
    %***Layer 2*******
    layer2_raw = layer1_out*w2 + repmat(b2',n_loop,1);
    
    %***Sigmoid******
    layer2_out = 1./(1+exp(-1*(layer2_raw))); %apply sigmoid
    
    net_labels(nstart:nend) = layer2_out;
    counter = counter + 1;
end

%% assign waveforms spike/noise labels
slabel = zeros(size(net_labels)); %spike classification
slabel(net_labels>=gamma) = 1;

spikes(:,2) = slabel;

%If applicable, modify NEV file
if strcmp(ext,'.nev') && writelabels

    % spikes and noise labels
    
    % move spikes to sort code 1 and noise to sort code 255
    sortcodes = slabel*1 + ~slabel*255;
    
    % which waveforms to overwrite
    spikesidx = ones(size(slabel));
    
    % set write=false for digital codes 
    % do the same for special channels which do not contain spiking data,
    % for example channels for uStim
    % (although moveToSortCode should already protect from doing that)
    spikesidx(spikes(:,1)==0)         = 0;
    sortcodes(spikes(:,1)==0)         = 0;
    spikesidx(spikes(:,1)>maxspikech) = 0;
    sortcodes(spikes(:,1)>maxspikech) = 0;
    
    moveToSortCode(filename,spikesidx,sortcodes,'channels',ch);

end
