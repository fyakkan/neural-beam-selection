base = '/Users/furkanyakkan/Documents/Next Generation Mobile Cominication Final/final/baseline';
nov  = '/Users/furkanyakkan/Documents/Next Generation Mobile Cominication Final/final/novel';
addpath(base, fullfile(base,'helpers'), nov);

D = loadBeamData(fullfile(base,'data'));            % 14 samples, step=5
NB = D.NumBeamPairs; nRx = D.prm.NumRxBeams; nTx = D.prm.NumTxBeams;   % 7 x 10
fprintf('grid %dx%d=%d  sampled=%d\n', nRx, nTx, NB, D.numSampled);

% 2D mask of observed beams (linear beam idx == column-major (Rx,Tx))
mask = zeros(nRx,nTx); mask(D.sampIdx) = 1;

mkimg = @(Tflat) reshape(Tflat.', nRx, nTx, 1, size(Tflat,1));   % N x 70 -> 7x10x1xN
to2ch = @(Timg,N) cat(3, Timg.*mask, repmat(mask,1,1,1,N));      % [observed | mask]

Ttr = mkimg(D.TtrainNN); Xtr = to2ch(Ttr, size(Ttr,4));
Tva = mkimg(D.TvalNN);   Xva = to2ch(Tva, size(Tva,4));
Tte = mkimg(D.TtestNN);  Xte = to2ch(Tte, size(Tte,4));

layers = [
    imageInputLayer([nRx nTx 2], Name="in", Normalization="none")
    convolution2dLayer(3, 64, Padding="same"); leakyReluLayer(0.01)
    convolution2dLayer(3, 64, Padding="same"); leakyReluLayer(0.01)
    convolution2dLayer(3, 64, Padding="same"); leakyReluLayer(0.01)
    convolution2dLayer(3, 64, Padding="same"); leakyReluLayer(0.01)
    convolution2dLayer(1, 1,  Padding="same")
    tanhLayer ];

rng(1);
dsTr = combine(arrayDatastore(Xtr,IterationDimension=4), arrayDatastore(Ttr,IterationDimension=4));
dsVa = combine(arrayDatastore(Xva,IterationDimension=4), arrayDatastore(Tva,IterationDimension=4));
opt = trainingOptions("adam", MaxEpochs=300, MiniBatchSize=512, InitialLearnRate=1e-3, ...
    LearnRateSchedule="piecewise", LearnRateDropFactor=0.5, LearnRateDropPeriod=80, ...
    ValidationData=dsVa, ValidationFrequency=50, OutputNetwork="best-validation-loss", ...
    ExecutionEnvironment="cpu", Plots="none", Verbose=false);
t0=tic; net = trainnet(dsTr, layers, "mse", opt); fprintf('CNN train %.1fs\n', toc(t0));

pred = predict(net, Xte);                       % 7x10x1xNte
predFlat = reshape(pred, NB, size(pred,4)).';   % Nte x 70
RR = evalTopK(predFlat, D);
fprintf('CNN: K@90=%s  acc@K=5=%.1f @8=%.1f @13=%.1f @20=%.1f @30=%.1f\n', ...
    num2str(RR.summary.K90_Neural), RR.accNeural(5), RR.accNeural(8), RR.accNeural(13), RR.accNeural(20), RR.accNeural(30));
fprintf('CNN RSRP@K=8=%.2f @13=%.2f  (optimal %.2f)\n', RR.rsrpNeural(8), RR.rsrpNeural(13), RR.summary.rsrp_optimal_dB);
fprintf('(baseline MLP was: K@90=13 acc@5=72.0 @8=81.4 @13=90.2 @20=94.2)\n');
disp('EXP_CNN_DONE');
