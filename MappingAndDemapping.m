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

% �����@ 4SPS���������ݣ�����һ�����������@1SPS�����ݣ������������������@ 4SPS�����ݣ�Ȼ���ֶ�downsample��@1SPS���ٽ��
% RetrievedData4SPS = modData4SPS;
% RetrievedData1SPS = downsample(RetrievedData4SPS, 4);

%% QAM demodulation
hDemodulator = comm.RectangularQAMDemodulator('ModulationOrder', 16, 'BitOutput', true, 'SymbolMapping', 'Gray');
RetrievedBinarySeq = step(hDemodulator, RetrievedData1SPS);

%% BER
hError = comm.ErrorRate;
errorStats = step(hError, DataPRBS, RetrievedBinarySeq);
BER = errorStats(1);