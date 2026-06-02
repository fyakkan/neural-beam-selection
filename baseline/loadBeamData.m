function D = loadBeamData(dataDir, cfg)
%loadBeamData  Load and preprocess the 5G NR beam-selection dataset.
%
%   D = loadBeamData(DATADIR, CFG) loads the official pre-recorded data
%   (nnBS_prm/TrainingData/TestData .mat) from DATADIR and builds the
%   inputs/targets for the REGRESSION baseline:
%       - NN input  : CFG.numSampled downsampled RSRP values (every 5th of
%                     the 70 beam pairs), in dB with -Inf floored.
%       - NN target : the full 70-element RSRP profile, min-max normalised
%                     to [-1,1] (matches the tanh output layer).
%   Beam-pair index == linear index into the 7x10 (Rx x Tx) RSRP matrix.
%
%   The returned struct D also carries the UE positions and true optimal
%   beam indices needed by the KNN / Statistical / Random / Exhaustive
%   benchmarks, and the ACTUAL (un-floored) test RSRP used for the
%   average-RSRP metric.
%
%   CFG fields (all optional):
%       .floorVal  (default -120)  value used to replace -Inf RSRP
%       .valFrac   (default 0.10)  fraction of training data held out for validation
%       .shuffleSeed (default 111) rng seed for the train/val split

    arguments
        dataDir (1,:) char
        cfg struct = struct()
    end
    if ~isfield(cfg,'floorVal'),    cfg.floorVal = -120;  end
    if ~isfield(cfg,'valFrac'),     cfg.valFrac  = 0.10;  end
    if ~isfield(cfg,'shuffleSeed'), cfg.shuffleSeed = 111; end
    if ~isfield(cfg,'sampleStep'),  cfg.sampleStep = 5;   end  % every 5th of 70 -> 14 beams

    S = load(fullfile(dataDir,'nnBS_prm.mat'));            prm = S.prm;
    tr = load(fullfile(dataDir,'nnBS_TrainingData.mat'));
    te = load(fullfile(dataDir,'nnBS_TestData.mat'));

    NumBeamPairs = prm.NumTxBeams * prm.NumRxBeams;        % 70
    if isfield(cfg,'sampIdx') && ~isempty(cfg.sampIdx)
        sampIdx = cfg.sampIdx(:).';                        % explicit (e.g. learned) selection
    else
        sampIdx = 1:cfg.sampleStep:NumBeamPairs;           % downsampled input beams (default 14)
    end

    % --- reshape RSRP cubes (Rx x Tx x N) -> (70 x N), linear beam index ---
    Rtr = double(reshape(tr.rsrpMatTrain, NumBeamPairs, []));   % 70 x Ntr
    Rte = double(reshape(te.rsrpMatTest,  NumBeamPairs, []));   % 70 x Nte
    Ntr = size(Rtr,2);   Nte = size(Rte,2);

    % --- floor -Inf (blocked beams) for NN input/target ---
    RtrF = Rtr;  RtrF(~isfinite(RtrF)) = cfg.floorVal;
    RteF = Rte;  RteF(~isfinite(RteF)) = cfg.floorVal;

    % --- target normalisation to [-1,1] using TRAINING stats (tanh output) ---
    lo = min(RtrF(:));  hi = max(RtrF(:));
    norm01 = @(x) 2*(x - lo)./(hi - lo) - 1;
    Ytr = norm01(RtrF);            % 70 x Ntr
    Yte = norm01(RteF);            % 70 x Nte  (kept for reference)

    % --- inputs: downsampled RSRP (floored dB; featureInputLayer z-scores) ---
    Xtr = RtrF(sampIdx, :);        % 14 x Ntr
    Xte = RteF(sampIdx, :);        % 14 x Nte

    % --- train / validation split (mirrors the official example) ---
    rng(cfg.shuffleSeed);
    perm   = randperm(Ntr);
    nVal   = round(cfg.valFrac * Ntr);
    valSel = perm(1:nVal);
    trnSel = perm(nVal+1:end);

    posTrainAll = tr.dataTrain.PosUE;        % Ntr x 3
    optTrainAll = tr.optBeamPairIdxTrain(:); % Ntr x 1

    % --- pack (observations as ROWS for featureInputLayer / trainnet) ---
    D = struct();
    D.prm          = prm;
    D.NumBeamPairs = NumBeamPairs;
    D.sampIdx      = sampIdx;
    D.numSampled   = numel(sampIdx);
    D.floorVal     = cfg.floorVal;
    D.normLo       = lo;   D.normHi = hi;

    % Neural-network train / val / test tensors
    D.XtrainNN = Xtr(:,trnSel).';   D.TtrainNN = Ytr(:,trnSel).';
    D.XvalNN   = Xtr(:,valSel).';   D.TvalNN   = Ytr(:,valSel).';
    D.XtestNN  = Xte.';             D.TtestNN  = Yte.';

    % Benchmarks (KNN uses position; same 90% split as NN training).
    % posTrain/posVal are aligned row-for-row with XtrainNN/XvalNN (same shuffle),
    % so they double as the position-modality inputs for the fusion network.
    D.posTrain  = posTrainAll(trnSel,:);     % Ntr90 x 3  (aligned with XtrainNN)
    D.posVal    = posTrainAll(valSel,:);     % Nval  x 3  (aligned with XvalNN)
    D.optTrain  = optTrainAll(trnSel);       % Ntr90 x 1  (true best beam per train UE)
    D.posTest   = te.dataTest.PosUE;         % Nte x 3   (aligned with XtestNN)
    D.optTest   = te.optBeamPairIdxTest(:);  % Nte x 1   (true best beam per test UE)

    % Actual test RSRP (UN-floored, -Inf preserved) for the RSRP metric
    D.rsrpTest  = Rte;                       % 70 x Nte

    D.NtrainNN = numel(trnSel);
    D.NvalNN   = numel(valSel);
    D.Ntest    = Nte;
end
