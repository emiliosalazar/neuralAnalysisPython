function nasSortCuedAttention(nevFile, nasNetBasename, gamma, wvLen, varargin)

p = inputParser;
p.addOptional('runNas',true,@islogical);
p.parse(varargin{:});

runNas = p.Results.runNas;

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
	sortDate = datestr(today);
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

nmMat = fullfile(pth, nmMat);

disp('Computing cued attention data format');
[dat, sortcodeLookupTable, rasterTimeline] = nev2datCuedAttn(nevFile);


preprocessDate = datestr(today);
save(nmMat, 'dat', 'sortcodeLookupTable', 'rasterTimeline', 'preprocessDate', 'sortDate');

disp('Preprocessing for data pipeline');
preprocessCuedAttn(nmMat, 'NasSort');
