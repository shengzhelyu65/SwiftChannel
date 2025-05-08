%%% SRS in 5G NR

%% System Parameters
clc
clear all
close all

numBSantennas = 64; % N_T
numUEantennas = 2; % N_R

numerology = 3; % miu

numRBs = 36;
numSubcarriersPerRB = 12;
numSymbolsPerSlot = 14;
subcarrierSpacing = 120; % 120 kHz

numFrames = 8*1; % 1 frame = 10 sub-frames
numSlotsPerSubframe = 2 ^ numerology; % N_slot
numSubframes = numFrames * 10; % N_sf
numSlots = numSubframes * numSlotsPerSubframe;

spatialCompressionRatios = [1, 2, 4, 8]; % N_T / N_RF
numRFChains = numBSantennas ./ spatialCompressionRatios;
frequencyCompressionRatios = [2, 4, 8, 16]; % the same as transmission comb nr

spatialCompressionRatiosIndexTr = 3;
frequencyCompressionRatiosIndexTr = 2;

SNRLevels = [6, 10, 14, 18, 22, 26, 30]; % in dB
UEVelocity = [5, 20, 40, 60, 80, 100, 120]; % in km/h

spatialCompressionRatio = spatialCompressionRatios(spatialCompressionRatiosIndexTr);
numRFChain = numRFChains(spatialCompressionRatiosIndexTr);
frequencyCompressionRatio = frequencyCompressionRatios(frequencyCompressionRatiosIndexTr);

numTestingSamples = 7*7*4; % number of total testing samples generated
folder = '<your_proj_folder_path>/Data/Testing';

%% UE and SRS Configuration
% Create UE/carrier configuration
ue = nrCarrierConfig;
ue.NSizeGrid = numRBs;      % Bandwidth in number of resource blocks (52RBs at 15kHz SCS for 10MHz BW)
ue.SubcarrierSpacing = subcarrierSpacing;  % 15, 30, 60, 120, 240 (kHz)
ue.CyclicPrefix = 'Normal'; % 'Normal' or 'Extended'

% Define SRS configuration
srs = nrSRSConfig;
srs.NumSRSSymbols = 1;            % Number of OFDM symbols allocated per slot
srs.SymbolStart = 13;             % Starting OFDM symbol within a slot
srs.NumSRSPorts = numUEantennas;  % Number of SRS antenna ports (1, 2, 4)
srs.FrequencyStart = 0;           % Frequency position of the SRS in BWP in RBs
srs.BSRS = 0;                     % Bandwidth configuration B_SRS (0...3). Controls allocated bandwidth to the SRS
srs.CSRS = 10;                    % Bandwidth configuration C_SRS (0...63). Controls allocated bandwidth to the SRS
srs.NRRC = 0;                     % Additional offset from FreqStart specified in blocks of 4 PRBs (0...67)
srs.BHop = 1;                     % Frequency hopping configuration (0...3). Set BHop < BSRS to enable frequency hopping
srs.KTC = 2;                      % Comb number (2, 4). Frequency density in subcarriers
srs.Repetition = 1;               % Repetition (1, 2, 4). Disables frequency hopping in blocks of |Repetition| symbols
srs.SRSPeriod = [numSlotsPerSubframe 0];            % Periodicity and offset in slots. SRSPeriod(2) must be < SRSPeriod(1)
srs.ResourceType = 'periodic';    % Resource type ('periodic', 'semi-persistent', 'aperiodic')

% duration = 1*srs.SRSPeriod(1); % Transmission length in slots
% hSRSGrid(ue, srs, duration, true);
% title('Carrier Grid Containing SRS')

% The logical variable |practicalSynchronization| controls channel
% synchronization behavior. When set to |true|, the example performs
% practical synchronization based on the values of the received SRS. When
% set to |false|, the example performs perfect synchronization.
% Synchronization is performed only in slots where the SRS is transmitted
% to keep perfect and practical channel estimates synchronized.
practicalSynchronization = true;

% Reset random generator for reproducibility
rng('default');

%% Processing Loop
% Total number of subcarriers and symbols per slot
K = numRBs * numSubcarriersPerRB; % numSubcarriers
L = numSymbolsPerSlot; % symbol_per_slot

% Initialize arrays storing channel estimates
slotGridSize = [K L numBSantennas numUEantennas];
hEst = zeros(slotGridSize);
hestInterp = zeros(slotGridSize);
hEstUpdate = zeros(slotGridSize);

% Initialize noise power estimate
nvar = 0;

% Calculate SRS CDM lengths
cdmLengths = hSRSCDMLengths(srs);

% OFDM symbols used for CSI acquisition
csiSelectSymbols = srs.SymbolStart + (1:srs.NumSRSSymbols);

totalGridSizeSymbolLevel = [K srs.NumSRSSymbols*numSubframes numBSantennas numUEantennas];
txGridSizeSymbolLevel = [K srs.NumSRSSymbols*numSubframes numUEantennas];
rxGridSizeSymbolLevel = [K srs.NumSRSSymbols*numSubframes numBSantennas];

channelModel = 'CDL-B';
carrierFreq = 28e9; % 28 GHz

cancelFlag = false;
for nChannel = 0:numTestingSamples-1
    if cancelFlag == true
        break
    end

    if nChannel > 0
        release(cdl);
    end
    
    nCombinations = length(UEVelocity) * length(SNRLevels);
    nChannelTmp = mod(nChannel, nCombinations);
    SNRLevelsIndexTr = fix(nChannelTmp/length(UEVelocity)) + 1;
    UEVelocityIndexTr = mod(nChannelTmp, length(UEVelocity)) + 1;
    SNRLevel = SNRLevels(SNRLevelsIndexTr);
    UEV = UEVelocity(UEVelocityIndexTr);
    
    fileEnding = "testing_SNR_" + SNRLevel + "_UEV_" + UEV + "_NFRA_" + numFrames +  "_DEL_B.mat";

    cdl = nrCDLChannel;
    cdl.DelayProfile = channelModel;
    cdl.CarrierFrequency = carrierFreq;
    
    cdl.TransmitAntennaArray.Size = [numUEantennas 1 1 1 1];
    cdl.ReceiveAntennaArray.Size = [numBSantennas 1 1 1 1];
    
    cdl.DelaySpread = 30e-9;
    c = physconst('lightspeed'); % speed of light in m/s
    cdl.MaximumDopplerShift = (UEV * 1000 / 3600) / c * carrierFreq; % UE max Doppler frequency in Hz
    cdl.Seed = nChannel;
    
    % Set channel sample rate
    ofdmInfo = nrOFDMInfo(ue);
    cdl.SampleRate = ofdmInfo.SampleRate;

    % Get the maximum delay of the channel
    chInfo = info(cdl);
    maxChDelay = chInfo.MaximumChannelDelay;
    
    % Initialize timing estimation offset 
    offset = chInfo.ChannelFilterDelay;

    hestPerfectSymbolLevel = zeros(totalGridSizeSymbolLevel);
    txGridSymbolLevel = zeros(txGridSizeSymbolLevel);
    lengthLast = ofdmInfo.SymbolLengths(end);
    rxWaveformSymbolLevel = zeros([numSubframes lengthLast numBSantennas]);

    f = waitbar(0,'1','Name',sprintf('Channel %d...', nChannel+1),'CreateCancelBtn','setappdata(gcbf,''canceling'',1)');
    setappdata(f,'canceling',0);

    for numSlot = 0:numSlots-1
        % Check for clicked Cancel button
        if getappdata(f,'canceling')
            cancelFlag = true;
            delete(f);
            break
        end
        waitbar(numSlot/numSlots, f, sprintf('Processing slot %d of %d...', numSlot+1, numSlots))

        nSlot = fix(numSlot / numSlotsPerSubframe);
        thisSlot = nSlot*srs.NumSRSSymbols + (1:srs.NumSRSSymbols);
        
        % Update slot counter
        ue.NSlot = numSlot;
        
        % Generate SRS and map to slot grid
        [srsIndices, srsIndInfo] = nrSRSIndices(ue, srs);
        srsSymbols = nrSRS(ue, srs);
        
        % Create a slot-wise resource grid empty grid and map SRS symbols
        txGrid = nrResourceGrid(ue, numUEantennas);
        txGrid(srsIndices) = srsSymbols;
        
        % Determine if the slot contains SRS
        isSRSSlot= ~isempty(srsSymbols);

        if isSRSSlot
            if mod(numSlot, numSlotsPerSubframe) ~= 0
                cancelFlag = true;
                delete(f);
                disp("ERROR!")
                break;
            end
        
            % OFDM Modulation
            [txWaveform, waveformInfo] = nrOFDMModulate(ue, txGrid);
            txWaveform = [txWaveform; zeros(maxChDelay, size(txWaveform, 2))];
        
            % Transmission through channel
            [rxWaveform, pathGains, sampleTimes] = cdl(txWaveform);
            
            % Add AWGN to the received time domain waveform
            % Normalize noise power to take account of sampling rate, which is
            % a function of the IFFT size used in OFDM modulation. The SNR
            % is defined per RE for each receive antenna (TS 38.101-4).
            SNR = SNRLevel;
            N0 = 1/sqrt(2.0 * numUEantennas * double(waveformInfo.Nfft) * SNR);
            noise = N0 * complex(randn(size(rxWaveform)), randn(size(rxWaveform)));
        
            rxWaveform = rxWaveform + noise;
            
            % Perform timing offset estimation 
            pathFilters = getPathFilters(cdl);
            
            % Timing estimation is only performed in the slots where the SRS is
            % transmitted to keep the perfect and practical channel estimation
            % synchronized.
            if isSRSSlot
                if practicalSynchronization
                    % Practical synchronization. Correlate the received waveform 
                    % with the SRS to give timing offset estimate
                    offset = nrTimingEstimate(ue, rxWaveform, srsIndices, srsSymbols);
                else
                    offset = nrPerfectTimingEstimate(pathGains, pathFilters);
                end
            end
        
            % Perform OFDM demodulation 
            rxWaveformDemo = rxWaveform(1+offset:end,:);
            rxGrid = nrOFDMDemodulate(ue, rxWaveformDemo);
            
            rxWaveformReMod = nrOFDMModulate(ue, rxGrid);
            rxWaveformDemoCut = rxWaveformReMod(end - lengthLast + 1:end,:);

            % plot txGrid and rxGrid
            % hTxRxGridPlot(txGrid, rxGrid);
        
            % Calculate perfect channel estimate for perfect PMI selection
            hEstPerfect = nrPerfectChannelEstimate(pathGains, pathFilters, numRBs, ...
                subcarrierSpacing, numSlot, offset, sampleTimes);
            
            % Save a copy of all transmitted OFDM grids and channel estimates
            hestPerfectSymbolLevel(:,thisSlot,:,:) = hEstPerfect(:,csiSelectSymbols,:,:);
            txGridSymbolLevel(:,thisSlot,:) = txGrid(:,csiSelectSymbols,:);
            rxWaveformSymbolLevel(nSlot+1,:,:) = rxWaveformDemoCut;
        end
    end

    % Save into file
    sampleIndex = nChannel;
    allHestPerfectFile = fullfile(folder, "hestPerfect_" + sampleIndex + "_" + fileEnding);
    allRxWaveformFile = fullfile(folder, "rxWaveform_" + sampleIndex + "_" + fileEnding);
    if nChannel == 0
        allTxGridFile = fullfile(folder, "txGrid_" + fileEnding);
    end

    % hOutputPlotPerFrame(allHestPerfect, allTxGrid, allRxGrid, 80);
    
    % Save the 4D matrix to MAT-file
    save(allHestPerfectFile, 'hestPerfectSymbolLevel', '-v7.3')
    save(allRxWaveformFile, 'rxWaveformSymbolLevel', '-v7.3');
    if nChannel == 0
        save(allTxGridFile, 'txGridSymbolLevel', '-v7.3');
    end

    delete(f);
end

disp("ALL DONE!")

%% Helper Functions
function hChannelEstimationPlot(symbols,subcarriers,allHestPerfect,allHestInterp)

    figure
    subplot(311)
    imagesc(symbols, subcarriers, abs(allHestPerfect(:,:,1,1))); 
    axis xy; xlabel('OFDM symbol'); ylabel('Subcarrier');
    colorbar;
    title('Perfect Channel Estimate (TxAnt=1, RxAnt=1)');

    subplot(312)
    imagesc(symbols, subcarriers, abs(allHestInterp(:,:,1,1)), ...
            'AlphaData',~isnan(allHestInterp(:,:,1,1))) 
    axis xy; xlabel('OFDM symbol'); ylabel('Subcarrier');
    colorbar;
    title('SRS-based Practical Channel Estimate (TxAnt=1, RxAnt=1) ');

    % Display channel estimation error, defined as the difference between the
    % SRS-based and perfect channel estimates
    subplot(313)
    hestErr = abs(allHestInterp - allHestPerfect);
    imagesc(symbols, subcarriers, hestErr(:,:,1,1),...
            'AlphaData',~isnan(hestErr(:,:,1,1)));
    axis xy; xlabel('OFDM symbol'); ylabel('Subcarrier');
    colorbar;
    title('Channel Estimation Error (TxAnt=1, RxAnt=1)');
    
end

% Displays the PMI evolution and PMI estimation SINR loss over time and
% frequency. The SINR loss is defined as a ratio of the SINR after
% precoding with estimated and perfect PMIs. Estimated PMIs are
% obtained using a practical channel estimate and perfect PMIs are
% selected using a perfect channel estimate.
function hPMIPlot(slots,resourceBlocks,pmiRB,pmiPerfectRB,lossRB)

    figure
    subplot(311)
    imagesc(slots,resourceBlocks,pmiPerfectRB,'AlphaData',~isnan(pmiPerfectRB)); axis xy;
    c = clim;
    cm = colormap;
    colormap( cm(1:floor(size(cm,1)/(c(2)-c(1)) -1):end,:) ); % Adjust colormap to PMI discrete values
    colorbar
    xlabel('Slot'); ylabel('Resource block'), title('PMI Selected using Perfect Channel Estimates')
    
    subplot(312)
    imagesc(slots,resourceBlocks,pmiRB,'AlphaData',~isnan(pmiRB)); axis xy;
    colorbar, 
    xlabel('Slot'); ylabel('Resource block'), title('PMI Selected using SRS')
    
    subplot(313)
    imagesc(slots,resourceBlocks,lossRB,'AlphaData',~isnan(lossRB));
    colormap(gca,cm)
    xlabel('Slot'); ylabel('Resource block'); axis xy; colorbar;
    title('PMI Estimation SINR Loss (dB)')  
    
end

% Displays the SINR per resource block obtained after precoding with the
% PMI that maximizes the SINR per subband.
function hBestSINRPlot(slots,resourceBlocks,sinrSubband,pmi,csiBandSize)

    % Display SINR after precoding with best PMI
    bestSINRPerSubband = nan(size(sinrSubband,[1 2]));

    % Get SINR per subband and slot using best PMI
    [sb,nslot] = find(~isnan(pmi));
    for i = 1:length(sb)
        bestSINRPerSubband(sb(i),nslot(i)) = sinrSubband(sb(i),nslot(i),pmi(sb(i),nslot(i))+1);
    end

    % First expand SINR from subbands into RBs for display purposes
    bestSINRPerRB = hExpandSubbandToRB(bestSINRPerSubband, csiBandSize, length(resourceBlocks));

    figure
    sinrdb = 10*log10(abs(bestSINRPerRB));
    imagesc(slots,resourceBlocks,sinrdb,'AlphaData',~isnan(sinrdb));
    axis xy; colorbar;
    xlabel('Slot');
    ylabel('Resource block')
    title('Average SINR Per Subband and Slot After Precoding with Best PMI (dB)')
    
end

% Expands a 2D matrix of values per subband in the first dimension into a
% matrix of values per resource block.
function rbValues = hExpandSubbandToRB(subbandValues, bandSize, NRB)

    lastBandSize = mod(NRB,bandSize);
    lastBandSize = lastBandSize + bandSize*(lastBandSize==0);

    rbValues = [kron(subbandValues(1:end-1,:),ones(bandSize,1));...
                subbandValues(end,:).*ones(lastBandSize,1)];
end

function [Grid,dispGrid] = hSRSGrid(carrier,srs,Duration,displayGrid,chplevels)
% [GRID,DISPGRID] = hSRSGrid(CARRIER,SRS,DURATION,DISPLAYGRID,CHPLEVELS)
% returns a multi-slot OFDM resource grid GRID containing a set of sounding
% reference signals in a carrier, as specified by the configuration objects
% CARRIER and SRS. This function also returns a scaled version of the grid
% used for display purposes. The optional input DURATION (Default 1)
% specifies the number of slots of the generated grid. The resource grid
% can be displayed using the optional input DISPLAYGRID (Default false).
% CHPLEVELS specifies the channel power levels for display purposes only
% and it must be of the same size as SRS.

    numSRS = length(srs);
    if nargin < 5
        chplevels = 1:-1/numSRS:1/numSRS;
        if nargin < 4
            displayGrid = false;
            if nargin < 3
                Duration = 1;
            end
        end
    end
    
    SymbolsPerSlot = carrier.SymbolsPerSlot;
    emptySlotGrid = nrResourceGrid(carrier,max([srs(:).NumSRSPorts])); % Initialize slot grid
    
    % Create the SRS symbols and indices and populate the grid with the SRS symbols
    Grid = repmat(emptySlotGrid,1,Duration);
    dispGrid = repmat(emptySlotGrid,1,Duration); % Frame-size grid for display
    for ns = 0:Duration-1
        slotGrid = emptySlotGrid;
        dispSlotGrid = emptySlotGrid; % Slot-size grid for display
        for ich = 1:numSRS
            srsIndices = nrSRSIndices(carrier,srs(ich));
            srsSymbols = nrSRS(carrier,srs(ich));
            slotGrid(srsIndices) = srsSymbols;
            dispSlotGrid(srsIndices) = chplevels(ich)*srsSymbols; % Scale the SRS for display only
        end
        OFDMSymIdx = ns*SymbolsPerSlot + (1:SymbolsPerSlot);
        Grid(:,OFDMSymIdx,:) = slotGrid;
        dispGrid(:,OFDMSymIdx,:) = dispSlotGrid;
        carrier.NSlot = carrier.NSlot+1;
    end
    
    if displayGrid
        plotGrid(dispGrid(:,:,1),chplevels,"SRS " + (1:numSRS)'); 
    end
end

function varargout = plotGrid(Grid,chplevels,leg)
% plotGrid(GRID, CHPLEVEL,LEG) displays a resource grid GRID containing
% channels or signals at different power levels CHPLEVEL and create a
% legend for these using a cell array of character vector LEG

    if nargin < 3
        leg = {'SRS'};
        if nargin < 2
            chplevels = 1;
        end
    end
    
    cmap = colormap(gcf);
    chpscale = length(cmap); % Scaling factor
    
    h = figure;
    image(0:size(Grid,2)-1,(0:size(Grid,1)-1)/12,chpscale*abs(Grid(:,:,1))); % Multiplied with scaling factor for better visualization
    axis xy;
    
    title('Carrier Grid Containing SRS')
    xlabel('OFDM Symbol'); ylabel('RB');
    
    clevels = chpscale*chplevels(:);
    N = length(clevels);
    L = line(ones(N),ones(N),'LineWidth',8); % Generate lines
    
    % Index the color map and associate the selected colors with the lines
    set(L,{'color'},mat2cell(cmap( min(1+fix(clevels),length(cmap) ),:),ones(1,N),3)); % Set the colors according to cmap
    
    % Create legend
    legend(leg(:));
    
    if nargout > 0 
        varargout = {h};
    end
end

function cdmLengths = hSRSCDMLengths(srs)

    % TS 38.211 Section 6.4.1.4.2, definition of N_ap_bar
    if (srs.EnableEightPortTDM)
        N_ap_bar = 4;
    else
        N_ap_bar = srs.NumSRSPorts;
    end

    % N_ap_bar = srs.NumSRSPorts;

    % TS 38.211 Section 6.4.1.4.3, position in comb per port
    halfPorts = (N_ap_bar==8 && srs.KTC==4) || ...
        (N_ap_bar==8 && srs.KTC==2 && srs.CyclicShift>=4) || ...
        (N_ap_bar==4 && srs.KTC==8) || ...
        (N_ap_bar==4 && srs.KTC==4 && srs.CyclicShift>=6) || ...
        (N_ap_bar==4 && srs.KTC==2 && srs.CyclicShift>=4);
    if halfPorts
        divfd = 2;
    elseif (N_ap_bar==8 && srs.KTC==8)
        divfd = 4;
    else
        divfd = 1;
    end

    F = N_ap_bar / divfd;

    cdmLengths = [F 1];

end

% Displays perfect and practical channel estimates and the channel
% estimation error for the first transmit and receive ports. The channel
% estimation error is defined as the absolute value of the difference
% between the perfect and practical channel estimates.
% Input: all 624, 14, 32, 4
function hChannelEstimationPlotForSlot(symbols,subcarriers,hest,hestInterp,hestPerfect)

    figure
    subplot(221)
    imagesc(symbols, subcarriers, abs(hest(:,:,1,1))); 
    axis xy; xlabel('OFDM symbol'); ylabel('Subcarrier');
    colorbar;
    title('Practical Channel Estimate on SRS positions (TxAnt=1, RxAnt=1)');

    subplot(222)
    imagesc(symbols, subcarriers, abs(hestInterp(:,:,1,1)), ...
            'AlphaData',~isnan(hestInterp(:,:,1,1))) 
    axis xy; xlabel('OFDM symbol'); ylabel('Subcarrier');
    colorbar;
    title('Interpolated Practical Channel Estimate (TxAnt=1, RxAnt=1) ');

    subplot(223)
    imagesc(symbols, subcarriers, abs(hestPerfect(:,:,1,1))); 
    axis xy; xlabel('OFDM symbol'); ylabel('Subcarrier');
    colorbar;
    title('Perfect Channel Estimate (TxAnt=1, RxAnt=1)');

    % Display channel estimation error, defined as the difference between the
    % SRS-based and perfect channel estimates
    subplot(224)
    hestErr = abs(hestInterp - hestPerfect);
    imagesc(symbols, subcarriers, hestErr(:,:,1,1),...
            'AlphaData',~isnan(hestErr(:,:,1,1)));
    axis xy; xlabel('OFDM symbol'); ylabel('Subcarrier');
    colorbar;
    title('Channel Estimation Error (TxAnt=1, RxAnt=1)');
end

function hRxGridPlotPerSlot(rxGrid, rxGridRF)

    figure
    subplot(121)
    surf(abs(squeeze(rxGrid(:,end,:))));
    shading('flat');
    xlabel('OFDM Symbols');
    ylabel('Subcarriers');
    zlabel('|H|');
    title('RxGrid (all BS antennas)');

    subplot(122)
    surf(abs(squeeze(rxGridRF(:,end,:))));
    shading('flat');
    xlabel('OFDM Symbols');
    ylabel('Subcarriers');
    zlabel('|H|');
    title('RxGrid (only RF chains)');
end

% Input:
% allHestPerfect: 624, 11200, 32, 4
% allRxGrid: 624, 11200, 32
function hOutputPlotPerFrame(allHestPerfect, allTxGrid, allRxGrid, NSlot)
    symbols = 0:NSlot*14-1;
    numSymbols = NSlot*14;
    subcarriers = 1:52*12;

    figure
    subplot(131)
    imagesc(symbols, subcarriers, abs(allHestPerfect(:,1:numSymbols,1,1))); 
    axis xy; xlabel('OFDM symbol'); ylabel('Subcarrier');
    colorbar;
    title('Perfect Channel Estimate (TxAnt=1, RxAnt=1)');

    subplot(132)
    imagesc(symbols, subcarriers, abs(allTxGrid(:,1:numSymbols,1))); 
    axis xy; xlabel('OFDM symbol'); ylabel('Subcarrier');
    colorbar;
    title('TxGrid (Tx1)');

    subplot(133)
    imagesc(symbols, subcarriers, abs(allRxGrid(:,1:numSymbols,1))); 
    axis xy; xlabel('OFDM symbol'); ylabel('Subcarrier');
    colorbar;
    title('RxGrid (Rx1)');
end

function hTxRxGridPlot(txGrid, rxGrid)
    figure
    subplot(121)
    imagesc(abs(txGrid(:,:,1)));
    xlabel('OFDM symbol'); ylabel('Subcarrier'); axis xy;
    title('Transmitted SRS');

    subplot(122)
    imagesc(abs(rxGrid(:,:,1)));
    xlabel('OFDM symbol'); ylabel('Subcarrier'); axis xy;
    title('Received SRS');
end

