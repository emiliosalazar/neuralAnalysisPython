function nmMat = nasSortGeneral(nevFile, nasNetBasename, gamma, wvLen, varargin)

if nargin < 4
	wvLen = 52;
	runNas = true;
	channelsGrab = 1:400;
	nmSuffix = '';
else p = inputParser;
	p.addOptional('runNas',true,@islogical);
	p.addOptional('channelsGrab',1:400,@isnumeric);
	p.addOptional('nmSuffix','',@ischar);
	p.parse(varargin{:});

	runNas = p.Results.runNas;
	channelsGrab = p.Results.channelsGrab;
	nmSuffix = p.Results.nmSuffix;
end

nevFileDirInfo = dir(nevFile);
nevFile = fullfile(nevFileDirInfo.folder, nevFileDirInfo.name);

if runNas
	sortDate = datestr(today);
	if iscell(nevFile)
		for fl = 1:length(nevFile)
			runNASNet(nevFile{fl}, gamma, nasNetBasename, wvLen, 'writelabels', true);
		end
	else
		runNASNet(nevFile, gamma, nasNetBasename, wvLen, 'writelabels', true);
	end
else
	nevInfo = dir(nevFile);
	sortDate = nevInfo.date;
end

if iscell(nevFile)
	fileInForName = nevFile{1};
else
	fileInForName = nevFile;
end

[pth, nm, ext] = fileparts(fileInForName);
% get rid of number after name
underscores = find(nm=='_');
nm(underscores(end):end) = [];

%if runNas
% NOTE be careful here because you've gotta know that this net DID sort the data if you're not sorting it heh
    [~, bsNas, ~] = fileparts(nasNetBasename);
	nmMat = [nm '_sorted-by_' bsNas];
%lse
%nmMat = nm;
%nd

nmMat = fullfile(pth, [nmMat, nmSuffix]);

fprintf('\nConverting nev to mat dat file\n');
dat = nev2dat(nevFile, 'channelsGrab', channelsGrab);


fprintf('Saving file to\n\t%s', nmMat)
preprocessDate = datestr(today);
save(nmMat, 'dat', 'preprocessDate', 'sortDate');

