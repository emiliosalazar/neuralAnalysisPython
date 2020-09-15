function nasSortCuedAttention(nevFile, nasNetBasename, gamma, wvLen, varargin)

p = inputParser;
p.addOptional('runNas',true,@islogical);
p.parse(varargin{:});

runNas = p.Results.runNas;

if runNas
	sortDate = today;
	if iscell(nevFile)
		for fl = 1:length(nevFile)
			runNASNet(nevFile{fl}, gamma, nasNetBasename, wvLen, 'writelabels', true)
		end
	else
		runNASNet(nevFile, gamma, nasNetBasename, wvLen, 'writelabels', true)
	end
else
	sortDate = [];
end

if iscell(nevFile)
	fileInForName = nevFile{1};
else
	fileInForName = nevFile
end

[pth, nm, ext] = fileparts(fileInForName);
% get rid of number after name
underscores = find(nm=='_');
nm(underscores(end):end) = [];

if runNas
	nmMat = [nm '_sorted-by_' natNetBasename];
else
	nmMat nm;
end

[dat, sortcodeLookupTable, rasterTimeline] = nev2datCuedAttn(nevFile);


preprocessDate = today;
save(nmMat, 'dat', 'sortcodeLookupTable', 'rasterTimeline', 'preprocessDate', 'sortDate');

preprocessCuedAttn(nmMat)
