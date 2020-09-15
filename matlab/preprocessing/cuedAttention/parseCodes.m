function out = parseCodes(in) 

splitStr = regexp(in,'(\w|\s|[= -]|\.)+(?=;).','match');                             %split up the string, could do 'split' option, but this keeps the semicolons
if any(cellfun(@numel,regexp(splitStr,'='))~=1) %there should be exactly one equals sign for each split string. -acs23feb2016
    warning('parseCodes:badString','Unable to evaluate encoded string ''%s'', which is possibly corrupt. Skipping...\n',splitStr{cellfun(@numel,regexp(splitStr,'='))~=1});
    splitStr(cellfun(@numel,regexp(splitStr,'='))~=1)=[]; %strip corrupted strings
end
splitStr = regexprep(splitStr,'((?<==)(((\d|\s|-)*\.?(\d|\s)*))+(?=;))','[$1]');     %bracket arrayable numbers
splitStr = regexprep(splitStr,'((?<==)([a-zA-Z._]|\s)+(?=;))','''$1''');             %put strings in single quotes (not sure whay I have a '.' in there...)
splitStr = regexprep(splitStr,'((?<==)rig\d[a-z]+(?=;))','''$1''');                  %put rig identifiers in single quotes -acs02dec2015
splitStr = regexprep(splitStr,'(?<=\=).((\d{1,3}\.){3}\d{1,3}).','''$1''');          %deal with the special case of something formatted like an IP address - put it in single quotes
splitStr = cellfun(@strcat,repmat({'out.'},size(splitStr)),splitStr,'uniformoutput',0);
try
    cellfun(@eval,splitStr,'uniformoutput',0);
catch  %#ok<CTCH>
    out = [];
    for s = 1:length(splitStr)
        try
            eval(splitStr{s});
        catch  %#ok<CTCH>
            warning('parseCodes:badString','Unable to evaluate encoded string ''%s'', which is possibly corrupt. Skipping...',splitStr{s});
        end
    end
end
