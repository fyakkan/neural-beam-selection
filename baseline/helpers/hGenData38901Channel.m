function [optBeamPairIdx,rsrpMat,dataInfo] = hGenData38901Channel(prm)
    % hGenData38901Channel Generate data for beam selection and beam
    % prediction examples

    %   Copyright 2024-2025 The MathWorks, Inc.

    rng(prm.Seed);                 % Set RNG state for repeatability
    c = physconst('LightSpeed');   % Propagation speed

    % Extract parameter fields that are needed later on in the function
    prm = validateParameters(prm);
    numTxBeams = prm.NumTxBeams;
    numRxBeams = prm.NumRxBeams;
    ncellid = prm.NCellID;
    txBurst = prm.TxBurst;
    ssbTransmittedBlocks = txBurst.TransmittedBlocks;
    rsrpMode = prm.RSRPMode;

    %% Burst Generation

    % Configure an nrDLCarrierConfig object to use the synchronization signal
    % burst parameters and to disable other channels. This object will be used
    % by nrWaveformGenerator to generate the SS burst waveform.
    cfgDL = configureWaveformGenerator(prm,txBurst);

    % Generate burst waveform
    burstWaveform = nrWaveformGenerator(cfgDL);
    burstWaveform = single(burstWaveform); % Convert to single for speedup

    %% Trajectory generation and 38.901 Channel Setup

    % Get carrier object
    carrier = nrCarrierConfig('NCellID',ncellid);
    carrier.NSizeGrid = cfgDL.SCSCarriers{1}.NSizeGrid;
    carrier.SubcarrierSpacing = cfgDL.SCSCarriers{1}.SubcarrierSpacing;

    % Get OFDM information
    ofdmInfo = nrOFDMInfo(carrier);
    sampleRate = ofdmInfo.SampleRate;

    % Compute the trajectories and the channels related to them. In case of
    % static UEs, UE locations are expressed as trajectories with a single
    % point at time 0.
    [trajectories,channels,dataInfo] = computeTrajectories(prm,sampleRate);

    %% Transmit-End Beam Sweeping
    % Transmit beam angles in azimuth and elevation, equi-spaced
    arrayTx = prm.TransmitAntennaArray;
    azBW = beamwidth(arrayTx,prm.CenterFrequency,'Cut','Azimuth');
    elBW = beamwidth(arrayTx,prm.CenterFrequency,'Cut','Elevation');
    txBeamAng = hGetBeamSweepAngles(numTxBeams,prm.TxAZlim,prm.TxELlim, ...
        azBW,elBW,prm.ElevationSweep);
    % Account for the antenna downtilt
    elOffset = 90 - prm.TxDowntilt;
    txBeamAng(2,:) = txBeamAng(2,:) + elOffset;

    % For evaluating transmit-side steering weights
    SteerVecTx = phased.SteeringVector(SensorArray=arrayTx,PropagationSpeed=c);

    % Get the set of OFDM symbols and subcarriers occupied by each SSB
    numBlocks = length(ssbTransmittedBlocks);
    burstStartSymbols = hSSBurstStartSymbols(txBurst.BlockPattern,numBlocks);
    burstStartSymbols = burstStartSymbols(ssbTransmittedBlocks==1);
    burstOccupiedSymbols = burstStartSymbols.' + (1:4);
    burstOccupiedSubcarriers = carrier.NSizeGrid*6 + (-119:120).';

    % Apply steering per OFDM symbol for each SSB
    gridSymLengths = repmat(ofdmInfo.SymbolLengths,1,cfgDL.NumSubframes);
    %   repeat burst over numTx to prepare for steering
    strTxWaveform = repmat(burstWaveform,1,prm.NumTx)./sqrt(prm.NumTx);
    wT = nan(prm.NumTx,numTxBeams);
    for txBeamIdx = 1:numTxBeams

        % Extract SSB waveform from burst
        blockSymbols = burstOccupiedSymbols(txBeamIdx,:);
        startSSBInd = sum(gridSymLengths(1:blockSymbols(1)-1))+1;
        endSSBInd = sum(gridSymLengths(1:blockSymbols(4)));
        ssbWaveform = strTxWaveform(startSSBInd:endSSBInd,1);

        % Generate weights for steered direction
        wT(:,txBeamIdx) = SteerVecTx(prm.CenterFrequency,txBeamAng(:,txBeamIdx));

        % Beamforming: Apply weights per transmit element to SSB
        strTxWaveform(startSSBInd:endSSBInd,:) = ssbWaveform*wT(:,txBeamIdx)';

    end

    % Adjust the beamformed waveform according to the base station power
    pref = sum(rms(strTxWaveform).^2);
    txWaveform = strTxWaveform*1/sqrt(pref)*sqrt(10^((prm.PowerBSs-30)/10));

    %% Receive-End Beam Sweeping and Measurement
    % Receive beam angles in azimuth and elevation, equi-spaced
    arrayRx = prm.ReceiveAntennaArray;
    azBW = beamwidth(arrayRx,prm.CenterFrequency,'Cut','Azimuth');
    elBW = beamwidth(arrayRx,prm.CenterFrequency,'Cut','Elevation');
    rxBeamAng = hGetBeamSweepAngles(numRxBeams,prm.RxAZlim,prm.RxELlim, ...
        azBW,elBW,prm.ElevationSweep);

    % For evaluating receive-side steering weights
    SteerVecRx = phased.SteeringVector(SensorArray=arrayRx,PropagationSpeed=c);
    wR = nan(prm.NumRx,numRxBeams);
    for rxBeamIdx = 1:numRxBeams
        wR(:,rxBeamIdx) = SteerVecRx(prm.CenterFrequency,rxBeamAng(:,rxBeamIdx));
    end

    %% Processing loop for each UE

    % The function loops over all receive locations to generate the data.
    % Note that, in this case, each separate location is represented as a
    % separate UE.
    numUEs = prm.NumTrajectories;
    ueNoiseFigure = prm.UENoiseFigure;
    posBS = dataInfo(1).PosBS;
    spatialConsistency = prm.ue.SpatialConsistency;
    isMobility = prm.ue.Mobility;

    % Pre-allocate outputs
    if isMobility
        % For mobility cases, the outputs are cell arrays with numUEs
        % elements
        rsrpMat_tmp = cell(numUEs,1); % This needs special handling to adhere to the parfor requirements
        optBeamPairIdx = cell(numUEs,1);
        % Create temporary variables to store fields of dataInfo structure
        % that change within the loop, as it's not allowed changing them
        % within a parfor loop
        los = cell(numUEs,1);
        txArrayOrientation = cell(numUEs,1);
        rxArrayOrientation = cell(numUEs,1);
    else
        rsrpMat = zeros(numRxBeams,numTxBeams,numUEs,"single");
        optBeamPairIdx = nan(numUEs,1);
    end

    % Get the maximum channel delay
    chInfo = arrayfun(@(x)info(x.SmallScale),channels);
    maxChDelay = max([chInfo(:).MaximumChannelDelay]);

    disp("  Total iterations: " + numUEs)
    % To enable the use of parallel computing for increased speed set the
    % value of |useParallel| below to true. This needs the Parallel
    % Computing Toolbox (TM). If this is not installed 'parfor' will
    % default to the normal 'for' statement.
    useParallel = false;
    parfor (ue = 1:numUEs, useParallel*numUEs)
        if mod(ue, 10)==1 || isMobility
            disp("  Iteration count = " + ue);
        end

        % Copy broadcast parameters to avoid extra parfor overhead
        wR_local = wR;
        thisChannel = channels(1,ue);
        thisTrajectory = trajectories(1,ue);
        if isMobility
            thisDataInfo = dataInfo(ue);
        else
            thisDataInfo = dataInfo; %#ok<PFBNS>
        end

        % Allocate output measurement variables for this trajectory
        numPoints = numel(thisTrajectory.Time);
        thisRsrpMat = zeros(numRxBeams,numTxBeams,numPoints,"single");
        thisOptBeamPairIdx = nan(numPoints,1);

        % Pad the waveform to ensure the channel filter is fully flushed
        nT = size(txWaveform,2);
        dlWaveform = [txWaveform; zeros(maxChDelay,nT)];

        % Loop over each time instance in the trajectory
        thisLOS = thisDataInfo.LOS;
        thisTxArrayOrientation = nan(numPoints,3);
        thisRxArrayOrientation = nan(numPoints,3);
        for tidx = 1:numPoints
            % Update channel according to trajectory up to the next "time of
            % interest"
            [thisChannel,thisLOS(tidx)] = updateChannel(thisChannel,thisTrajectory,thisDataInfo.Outdoor,thisDataInfo.LOS(tidx),tidx,spatialConsistency,posBS);

            if isMobility
                % Update BS and UE orientation at this point
                thisTxArrayOrientation(tidx,:) = thisChannel.SmallScale.TransmitArrayOrientation(:)';
                thisRxArrayOrientation(tidx,:) = thisChannel.SmallScale.ReceiveArrayOrientation(:)';
            end

            % Pass the waveform through the channel
            rxWaveform = thisChannel.SmallScale(dlWaveform);
            rxWaveform = rxWaveform*db2mag(thisChannel.LargeScale(posBS,thisTrajectory.Position(tidx,:))); % Account for the path loss

            % Apply AWGN
            rxWaveform = hAWGN(rxWaveform,ueNoiseFigure,sampleRate);

            % Loop over all receive beams
            rsrp = -inf(numRxBeams,numTxBeams);
            for rIdx = 1:numRxBeams

                % Beam combining: Apply weights per receive element
                strRxWaveform = rxWaveform*conj(wR_local(:,rIdx));

                % Correct timing
                offset = hSSBurstTimingOffset(strRxWaveform,carrier,ofdmInfo,burstOccupiedSymbols);
                if offset > maxChDelay
                    % If the receiver cannot compute a valid timing offset, the
                    % receive power of the waveform is too low. Continue to the
                    % next receive beam.
                    continue
                end
                strRxWaveformS = strRxWaveform(1+offset:end,:);

                % OFDM Demodulate
                rxGrid = nrOFDMDemodulate(carrier,strRxWaveformS);

                % Loop over all SSBs in rxGrid (transmit end)
                for tIdx = 1:numTxBeams
                    % Get each SSB grid
                    rxSSBGrid = rxGrid(burstOccupiedSubcarriers, ...
                        burstOccupiedSymbols(tIdx,:),:);

                    % Compute the synchronization signal RSRP
                    rsrp(rIdx,tIdx) = hSSBurstRSRP(rxSSBGrid,ncellid,ssbTransmittedBlocks,tIdx,rsrpMode);
                end
            end
            % Assign the RSRP value to the output matrix after converting it to
            % single to avoid memory waste
            thisRsrpMat(:,:,tidx) = single(rsrp);

            %% Beam Determination
            [~,optBeamIdx] = max(rsrp,[],'all','linear'); % First occurrence is output
            thisOptBeamPairIdx(tidx) = optBeamIdx;
        end

        % Assign values that will need to be plugged back into the dataInfo
        % structure after the for loop
        if isMobility
            los{ue} = thisLOS;
            txArrayOrientation{ue} = thisTxArrayOrientation;
            rxArrayOrientation{ue} = thisRxArrayOrientation;
        end

        % Assign the measurements results for this trajectory to the output
        % variables
        if isMobility
            rsrpMat_tmp{ue} = thisRsrpMat;
            optBeamPairIdx{ue} = thisOptBeamPairIdx;
        else
            rsrpMat(:,:,ue) = thisRsrpMat;
            optBeamPairIdx(ue) = thisOptBeamPairIdx;
        end
    end

    if isMobility
        rsrpMat = rsrpMat_tmp;
        % Re-assign the fields of dataInfo that have changed within the
        % loop
        [dataInfo.LOS] = los{:};
        [dataInfo.TransmitArrayOrientation] = txArrayOrientation{:};
        [dataInfo.ReceiveArrayOrientation] = rxArrayOrientation{:};
    end
end

%% Local Functions
function prm = validateParameters(prm)
    % Check whether the input parameters are related to a mobility-based
    % simulation (i.e., time-domain beam prediction) or not (i.e.,
    % spatial-domain beam prediction)
    prm.ue.Mobility = isfield(prm,"ue"); % For mobility-based simulations, the input parameter structure must have a field called "ue"
    if ~prm.ue.Mobility
        % Update prm to add mobility parameters, even though the mobility
        % is zero. This will make the overall code flow
        prm.NumTrajectories = prm.NumUELocations;
        prm.ue.MinDistance2D = 0; % meters
        prm.ue.Speed = 0; % km/h
        prm.ue.RotationSpeed = 0; % RPM
        prm.ue.MaxTrajectoryDuration = 0; % s
        prm.ue.MinTrajectoryDuration = 0; % s
        prm.ue.TimeStep = 0.1; % Nonzero time step, to avoid empty trajectory time, in seconds
        prm.ue.SpatialConsistency = true; % Static spatial consistency
    end
end

function cfgDL = configureWaveformGenerator(prm,txBurst)
    % Configure an nrDLCarrierConfig object to be used by nrWaveformGenerator
    % to generate the SS burst waveform.

    % Calculate the minimum number of subframes for the given number of
    % transmitted blocks to avoid generating a waveform that is longer than
    % needed
    carrier = nrCarrierConfig(SubcarrierSpacing=prm.SCS);
    symbolsPerSubframe = carrier.SymbolsPerSlot*carrier.SlotsPerSubframe;
    numBlocks = length(txBurst.TransmittedBlocks);
    burstStartSymbols = hSSBurstStartSymbols(txBurst.BlockPattern,numBlocks);
    burstStartSymbols = burstStartSymbols(txBurst.TransmittedBlocks==1);
    burstOccupiedSymbols = burstStartSymbols.' + (1:4);
    numSubframes = ceil(burstOccupiedSymbols(prm.NumSSBlocks,end)/symbolsPerSubframe);

    % For mobility-based simulations, ensure that the waveform is shorter
    % than the time step used to advance the trajectory
    if prm.ue.Mobility && (numSubframes*1e-3 > prm.ue.TimeStep)
        error("Time step used for trajectory generation (" + prm.ue.TimeStep + ...
            "s) must be greater than the waveform length (" + ...
            numSubframes*1e-3 + "s).");
    end

    cfgDL = nrDLCarrierConfig;
    cfgDL.SCSCarriers{1}.SubcarrierSpacing = prm.SCS;
    cfgDL.SCSCarriers{1}.NSizeGrid = 20; % Make the grid as tight as possible around the SSB for speedup
    if (prm.SCS==240)
        cfgDL.SCSCarriers = [cfgDL.SCSCarriers cfgDL.SCSCarriers];
        cfgDL.SCSCarriers{2}.SubcarrierSpacing = prm.SubcarrierSpacingCommon;
        cfgDL.BandwidthParts{1}.SubcarrierSpacing = prm.SubcarrierSpacingCommon;
    else
        cfgDL.BandwidthParts{1}.SubcarrierSpacing = prm.SCS;
    end
    cfgDL.BandwidthParts{1}.NSizeBWP = cfgDL.SCSCarriers{1}.NSizeGrid;
    cfgDL.PDSCH{1}.Enable = false;
    cfgDL.PDCCH{1}.Enable = false;
    cfgDL.ChannelBandwidth = prm.ChannelBandwidth;
    cfgDL.FrequencyRange = prm.FrequencyRange;
    cfgDL.NCellID = prm.NCellID;
    cfgDL.NumSubframes = numSubframes;
    cfgDL.WindowingPercent = 0;
    cfgDL.SSBurst = txBurst;

end

function rxWaveform = hAWGN(rxWaveform,noiseFigure,sampleRate)
    % Add noise to the received waveform

    persistent kBoltz;
    if isempty(kBoltz)
        kBoltz = physconst('Boltzmann');
    end

    % Calculate the required noise power spectral density
    NF = 10^(noiseFigure/10);
    N0 = sqrt(kBoltz*sampleRate*290*NF);

    % Establish dimensionality based on the received waveform
    [T,Nr] = size(rxWaveform);

    % Create noise
    noise = N0*randn([T Nr],'like',1i);

    % Add noise to the received waveform
    rxWaveform = rxWaveform + noise;
end

function [ch,los] = updateChannel(ch,traj,isOutdoor,los,tidx,spatialConsistency,posBS)
    % Update channel according to trajectory up to the next "time of interest"

    if any(strcmpi(spatialConsistency,{'ProcedureA','ProcedureB'}))
        d_step = 1; % Distance for spatially-consistent channel updates in meters
        omega = traj.RotationSpeed; % Rotational speed in RPM
        pos = traj.Position(tidx,:);
        vel = traj.VelocityDirection(tidx,:)./norm(traj.VelocityDirection(tidx,:))*traj.Speed;
        cfg = struct(SpatialConsistency=spatialConsistency,UpdateDistance=d_step);
        BS = struct(Position=posBS,Velocity=[0 0 0],RotationVelocity=[0; 0; 0]); % BS does not move
        UE = struct(Position=pos,Velocity=vel,RotationVelocity=[omega; omega; omega]);

        % Apply spatial consistency update
        if (tidx==1)
            h38901Channel.createChannelLink(setfield(ch.ChannelConfiguration,SitePositions=BS.Position));
        end
        ch = h38901Channel.spatiallyConsistentMobility(ch,cfg,traj.Time(tidx),BS,UE); % this function also updates ch.SmallScale.InitialTime

        % Update LOS value
        los = ch.SmallScale.HasLOSCluster && isOutdoor;
    end

end

function [trajectories,channels,data] = computeTrajectories(prm,sampleRate)

    % Extract parameter fields that are needed later on in the function
    ISD = prm.InterSiteDistance; % m
    numTrajectories = prm.NumTrajectories;
    tmax = prm.ue.MaxTrajectoryDuration; % s
    dt = prm.ue.TimeStep; % s
    v = prm.ue.Speed; % km/h
    v = v*1e3/3600; % m/s
    omega = prm.ue.RotationSpeed; % RPM
    spatialConsistency = prm.ue.SpatialConsistency;

    % Define parameters needed to compute 38.901 scenario and trajectories
    trajectoryTemplate = struct(Speed=v,RotationSpeed=omega,...
        Time=double.empty,Position=double.empty(0,3),VelocityDirection=double.empty(0,1));
    trajectories = repmat(trajectoryTemplate,1,0);
    dataTemplate = struct(NumTrajectories=prm.NumTrajectories,Seed=prm.Seed,PosBS=double.empty(0,3),PosUE=double.empty(0,3),...
        Outdoor=double.empty(0,1),LOS=double.empty(0,1),...
        TransmitArrayOrientation=double.empty(0,3),ReceiveArrayOrientation=double.empty(0,3));
    trajectoryTime = (0:dt:tmax)'; % Column vector
    dr = v*trajectoryTime;
    if prm.ue.Mobility
        data = repmat(dataTemplate,1,numTrajectories);
        alpha = 360*rand(numTrajectories,1); % Random direction of travel, deg
    else
        data = dataTemplate;
        data.NumUELocations = data.NumTrajectories;
        % alpha is needed but only used for mobility simulations. Setting
        % it here to a placeholder value of the right size to avoid
        % changing the RNG value when a call to rand() is not needed.
        alpha = zeros(numTrajectories,1);
    end
    if prm.ue.Mobility
        % In the mobility scenario, all UEs are outdoor
        indoorRatio = 0;
    else
        % If the UEs are static, their indoor/outdoor position is
        % determined using TR 38.901 Tables 7.2-1 and 7.2-3.
        indoorRatio = [];
    end

    % Get the cell boundaries
    [sitex,sitey] = h38901Channel.sitePolygon(ISD);

    % Generate the 38901 scenario class
    s38901 = h38901Scenario(Scenario=prm.Scenario,...
        IndoorRatio=indoorRatio,...
        CarrierFrequency=prm.CenterFrequency,...
        InterSiteDistance=prm.InterSiteDistance,...
        NumCellSites=1,...
        NumSectors=3,...
        NumUEs=numTrajectories,...
        ChosenUEs=true,...
        SpatialConsistency=spatialConsistency,...
        Wrapping=false,...
        Seed=prm.Seed);

    % Generate the trajectories
    channels = [];
    NTraj = 0;
    while NTraj<prm.NumTrajectories
        [trajectories,channels,data,NTraj] = generateMultipleTrajectories(s38901,trajectories,channels,data,NTraj,... % Inputs used and modified in the output
            prm,trajectoryTemplate,trajectoryTime,dr,alpha,... % Input needed for trajectory generation
            sampleRate,sitex,sitey); % Inputs needed for 38.901 channel generation
    end
end

function [trajectories,channels,data,NTraj] = generateMultipleTrajectories(s38901,trajectories,channels,data,NTraj,... % Inputs used and modified in the output
        prm,trajectoryTemplate,trajectoryTime,dr,alpha,... % Input needed for trajectory generation
        sampleRate,sitex,sitey) % Inputs needed for 38.901 channel generation

    % Define the number of trajectories needed
    numTraj = prm.NumTrajectories-NTraj;
    s38901.NumUEs = numTraj;

    % Generate the channels and get the UEs positions
    [thisChannels,~,thisData] = h38901ChannelSetup(s38901,prm,sampleRate);

    % Get BS and UE initial positions
    posBS = thisData.PosBS; % m
    posUE_start = thisData.PosUE; % m

    thisTrajectories = repmat(trajectoryTemplate,1,numTraj);
    for n = 1:numTraj
        % Generate the trajectory for the full time
        [thisTime,thisPos,thisVel]= generateSingleTrajectory(prm,posUE_start(n,:),alpha(n),dr,trajectoryTime,posBS,sitex,sitey);

        if isempty(thisTime)
            % If the trajectory time is too small, discard it
            continue;
        else
            % Assign the output trajectory
            thisTrajectories(n).Time = thisTime;
            thisTrajectories(n).Position = thisPos;
            thisTrajectories(n).VelocityDirection = thisVel;

            NTraj = NTraj + 1;

            if prm.ue.Mobility
                % Update data output
                data(NTraj).Trajectory = thisTrajectories(n);
                data(NTraj).PosBS = posBS;
                data(NTraj).PosUE = thisPos;
                data(NTraj).Outdoor = thisData.Outdoor(n);
                data(NTraj).LOS = repmat(thisData.LOS(n),numel(thisTime),1);
                data(NTraj).TransmitArrayOrientation = thisData.TransmitArrayOrientation(n,:);
                data(NTraj).ReceiveArrayOrientation = thisData.ReceiveArrayOrientation(n,:);
            end
        end
    end

    % Remove all invalid trajectories and add the new data to the previous loop
    dataToRemove = arrayfun(@(x)isempty(x.Time),thisTrajectories);
    thisTrajectories(dataToRemove) = [];
    thisChannels(dataToRemove) = [];
    trajectories = cat(2,trajectories,thisTrajectories);
    channels = cat(2,channels,thisChannels(1,:));

    if prm.ue.Mobility
        % For mobility simulations, display a log with progress information
        % to the user
        disp("  " + (numTraj-nnz(dataToRemove)) + "/" + numTraj + " trajectories generated.");
    else
        data.PosUE = cat(1,data.PosUE,thisData.PosUE);
        data.Outdoor = cat(1,data.Outdoor,thisData.Outdoor(:));
        data.LOS = cat(1,data.LOS,thisData.LOS(:));
        data.TransmitArrayOrientation = cat(1,data.TransmitArrayOrientation,thisData.TransmitArrayOrientation);
        data.ReceiveArrayOrientation = cat(1,data.ReceiveArrayOrientation,thisData.ReceiveArrayOrientation);
        if isempty(data.PosBS)
            data.PosBS = posBS;
        end
    end

end

function [thisTime,thisPos,thisVel] = generateSingleTrajectory(prm,posUE_start,alpha,dr,trajectoryTime,posBS,sitex,sitey)
    % Generate the trajectory for the full time

    if ~prm.ue.Mobility
        thisTime = 0;
        thisPos = posUE_start;
        thisVel = [0 0 0];
    else
        tmin = prm.ue.MinTrajectoryDuration; % s
        min_d_2D = prm.ue.MinDistance2D; % meters

        thisTime = trajectoryTime;
        thisVel = repmat([cosd(alpha), sind(alpha), 0],numel(thisTime),1); % Constant velocity direction
        thisPos = posUE_start + dr.*thisVel;

        % Remove all points that lie outside the sector boundaries
        idx = vecnorm(thisPos,2,2)<min_d_2D | ... % not greater than the minimum allowed
            thisPos(:,1)<0 | atand(thisPos(:,2)./thisPos(:,1))<-30 | atand(thisPos(:,2)./thisPos(:,1))>90 | ... % not in the first sector
            ~inpolygon(thisPos(:,1),thisPos(:,2),sitex + posBS(1,1),sitey + posBS(1,2));
        thisTime(idx) = [];
        thisPos(idx,:) = [];
        thisVel(idx,:) = [];

        % If the trajectory time is too small, discard it
        if ~isempty(thisTime) && thisTime(end)<tmin
            thisTime = [];
            thisPos = [];
            thisVel = [];
        end
    end

end

function [channels,chInfo,dataInfo] = h38901ChannelSetup(s38901,prm,sampleRate)
    % Generate channels compliant with TR 38.901 using the scenario parameters

    % Create 38.901-compliant channels between the first sector of a
    % three-sector node and all the UEs randomly dropped in the sector
    [channels,chinfoAll] = createChannelLinks(s38901,...
        SampleRate=sampleRate,...
        DropMode="PathLoss",...
        TransmitAntennaArray=hPhasedToNRArray(prm.TransmitAntennaArray,prm.Lambda),...
        ReceiveAntennaArray=hPhasedToNRArray(prm.ReceiveAntennaArray,prm.Lambda),...
        FastFading=true,...
        EvaluatePathLoss=false,...
        Site=1,...
        Sector=1);
    chInfo = chinfoAll.AttachedUEInfo;

    % Ensure channel filtering is set to true to be able to pass the
    % waveform through the channel
    for ch = 1:numel(channels)
        channels(ch).SmallScale.ChannelFiltering = true;

        % Add transmit and receive array orientation info to the output data
        % structure
        dataInfo.TransmitArrayOrientation(ch,:) = channels(ch).SmallScale.TransmitArrayOrientation';
        dataInfo.ReceiveArrayOrientation(ch,:) = channels(ch).SmallScale.ReceiveArrayOrientation';
    end

    % Add the number of UEs and seed to the output data structure
    dataInfo.NumUELocations = prm.NumTrajectories;
    dataInfo.Seed = prm.Seed;

    % Add the UE and BS positions to the output data structure, together with
    % info on whether the UE is in line of sight or not
    dataInfo.PosBS = chInfo(1).Config.BSPosition;
    dataInfo.PosUE = cat(1,chInfo.Position);
    dataInfo.Outdoor = [chInfo.d_2D_in]==0;
    % To be in perfect line of sight, the UE must be outdoor as well.
    % Note that the same information is contained in
    % channels(idx).SmallScale.HasLOSCluster
    dataInfo.LOS = [chInfo.LOS] & dataInfo.Outdoor;
end