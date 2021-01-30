function codesToMat

exGlobals

[filepath,~,~] = fileparts(mfilename('fullpath'));
fileSave = fullfile(filepath, 'trialcodes');
codesStruct = codes;
codesArray(struct2array(codes)) = fieldnames(codes);
save(fileSave, 'codesArray', 'codesStruct')