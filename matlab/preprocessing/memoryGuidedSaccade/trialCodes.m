function codes = trialCodes()
% trial code dictionary
codes.START_TRIAL = 1;
codes.ALIGN = 253; % use if you want to align your data to a certain code
codes.SHOWEX_ABORT = 254;
codes.END_TRIAL = 255;

% stimulus / trial event codes
codes.FIX_ON = 2 ;
codes.FIX_OFF = 3 ;
codes.FIX_MOVE = 4 ;
codes.REWARD = 5 ;
codes.DIODE_ON = 6 ;
codes.DIODE_OFF = 7 ;
%
codes.NO_STIM = 9;
codes.STIM_ON = 10 ;
codes.STIM1_ON = 11 ;
codes.STIM2_ON = 12 ;
codes.STIM3_ON = 13 ;
codes.STIM4_ON = 14 ;
codes.STIM5_ON = 15 ;
codes.STIM6_ON = 16 ;
codes.STIM7_ON = 17 ;
codes.STIM8_ON = 18 ;
codes.STIM9_ON = 19 ;
codes.STIM10_ON = 20 ;
codes.STIM_OFF = 40 ;
codes.STIM1_OFF = 41 ;
codes.STIM2_OFF = 42 ;
codes.STIM3_OFF = 43 ;
codes.STIM4_OFF = 44 ;
codes.STIM5_OFF = 45 ;
codes.STIM6_OFF = 46 ;
codes.STIM7_OFF = 47 ;
codes.STIM8_OFF = 48 ;
codes.STIM9_OFF = 49 ;
codes.STIM10_OFF = 50 ;
codes.TARG_ON = 70 ;
codes.TARG1_ON = 71 ;
codes.TARG2_ON = 72 ;
codes.TARG3_ON = 73 ;
codes.TARG4_ON = 74 ;
codes.TARG5_ON = 75 ;
codes.TARG6_ON = 76 ;
codes.TARG7_ON = 77 ;
codes.TARG8_ON = 78 ;
codes.TARG9_ON = 79 ;
codes.TARG10_ON = 80 ;
codes.TARG_OFF = 100 ;
codes.TARG1_OFF = 101 ;
codes.TARG2_OFF = 102 ;
codes.TARG3_OFF = 103 ;
codes.TARG4_OFF = 104 ;
codes.TARG5_OFF = 105 ;
codes.TARG6_OFF = 106 ;
codes.TARG7_OFF = 107 ;
codes.TARG8_OFF = 108 ;
codes.TARG9_OFF = 109 ;
codes.TARG10_OFF = 110 ;
codes.CHOICE0 = 120;
codes.CHOICE1 = 121;
codes.CHOICE2 = 122;
codes.CHOICE3 = 123;
codes.CHOICE4 = 124;
codes.CHOICE5 = 125;
codes.CHOICE6 = 126;
codes.CHOICE7 = 127;
codes.CHOICE8 = 128;
codes.CHOICE9 = 129;

% ustim codes
codes.USTIM_ON = 130 ;
codes.USTIM_OFF = 131 ;

% sound codes
codes.SOUND_ON = 132 ;
codes.SOUND_OFF = 133 ;
codes.SOUND_CHANGE = 134;

% behavior codes
codes.FIXATE  = 140 ;	% attained fixation 
codes.SACCADE = 141 ;	% initiated saccade

% trial outcome codes
codes.CORRECT = 150 ;	% Independent of whether reward is given
codes.IGNORED = 151 ;	% Never fixated or started trial
codes.BROKE_FIX = 152 ; % Left fixation before trial complete
codes.WRONG_TARG = 153 ; % Chose wrong target
codes.BROKE_TARG = 154 ; % Left target fixation before required time
codes.MISSED = 155 ;	% for a detection task
codes.FALSEALARM = 156 ;
codes.NO_CHOICE = 157 ;	% saccade to non-target / failure to leave fix window
codes.WITHHOLD = 158 ; %correctly-withheld response
codes.ACQUIRE_TARG = 159; % Acquired the target
codes.FALSE_START = 160 ; % left too early
codes.BCI_CORRECT = 161 ; % BCI task (vs. non-BCI behavior) performed correct
codes.BCI_MISSED = 162 ; % BCI task (vs. non-BCI behavior) performed incorrectly 
codes.CORRECT_REJECT = 163;
codes.LATE_CHOICE = 164;
