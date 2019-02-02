clear; close all;

%% Global parameters
M = 16; % 16QAM
DataNum = 204800;

%% Data in
rng(6);
DataPRBS = randi([0 1], DataNum*log2(M), 1); % ltePRBS(6, DataNum*log2(M))

%% QAM modulation
hModulator = comm.RectangularQAMModulator('ModulationOrder', 16, 'BitInput', true, 'SymbolMapping', 'Gray');
modData1SPS = step(hModulator, DataPRBS);
modData4SPS = upsample(modData1SPS, 4);

%% Channel
RetrievedData1SPS = modData1SPS;

% 如果是@ 4SPS采样的数据，方法一：让网络输出@1SPS的数据；方法二：让网络输出@ 4SPS的数据，然后手动downsample到@1SPS，再解调
% RetrievedData4SPS = modData4SPS;
% RetrievedData1SPS = downsample(RetrievedData4SPS, 4);

%% QAM demodulation
hDemodulator = comm.RectangularQAMDemodulator('ModulationOrder', 16, 'BitOutput', true, 'SymbolMapping', 'Gray');
RetrievedBinarySeq = step(hDemodulator, RetrievedData1SPS);

%% BER
hError = comm.ErrorRate;
errorStats = step(hError, DataPRBS, RetrievedBinarySeq);
BER = errorStats(1);