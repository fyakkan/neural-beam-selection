% Baseline ablations for Task 6: effect of network width/depth and UE density
% (training-set size) on the RSRP-only regression beam selector (clean RSRP).
base = '/Users/furkanyakkan/Documents/Next Generation Mobile Cominication Final/final/baseline';
nov  = '/Users/furkanyakkan/Documents/Next Generation Mobile Cominication Final/final/novel';
addpath(base, fullfile(base,'helpers'), nov);
D = loadBeamData(fullfile(base,'data'));
NB=D.NumBeamPairs;

opt = @() trainingOptions("adam", MaxEpochs=300, MiniBatchSize=512, InitialLearnRate=1e-3, ...
    LearnRateSchedule="piecewise", LearnRateDropFactor=0.5, LearnRateDropPeriod=80, ...
    ValidationData={D.XvalNN,D.TvalNN}, ValidationFrequency=50, OutputNetwork="best-validation-loss", ...
    ExecutionEnvironment="cpu", Plots="none", Verbose=false);

% ---- (a) network width sweep (4 hidden layers) ----
fprintf('\n[A] Effect of hidden width (4-layer MLP, RSRP-only, clean):\n');
fprintf('%6s | %-6s | %-8s | %-8s\n','width','K@90','acc@K13','acc@K8');
for w=[32 48 96 192 384]
    rng(1);
    net = trainnet(D.XtrainNN, D.TtrainNN, buildRegressionNet(D.numSampled,NB,w), "mse", opt());
    R = evalTopK(predict(net,D.XtestNN), D);
    fprintf('%6d | %4s | %7.1f  | %7.1f\n', w, num2str(R.summary.K90_Neural), R.accNeural(13), R.accNeural(8));
end

% ---- (b) network depth sweep (width 96) ----
fprintf('\n[B] Effect of depth (#hidden layers, width 96, RSRP-only, clean):\n');
fprintf('%6s | %-6s | %-8s | %-8s\n','layers','K@90','acc@K13','acc@K8');
for nl=[2 3 4 6 8]
    rng(1);
    net = trainnet(D.XtrainNN, D.TtrainNN, mlp(D.numSampled,NB,96,nl), "mse", opt());
    R = evalTopK(predict(net,D.XtestNN), D);
    fprintf('%6d | %4s | %7.1f  | %7.1f\n', nl, num2str(R.summary.K90_Neural), R.accNeural(13), R.accNeural(8));
end

% ---- (c) UE density (training-set fraction) ----
fprintf('\n[C] Effect of UE density (train-set fraction, width 96, RSRP-only, clean):\n');
fprintf('%8s | %-7s | %-6s | %-8s\n','frac','Ntrain','K@90','acc@K13');
Ntr = size(D.XtrainNN,1);
for f=[0.05 0.1 0.25 0.5 1.0]
    rng(1); m=round(f*Ntr); sel=randperm(Ntr,m);
    net = trainnet(D.XtrainNN(sel,:), D.TtrainNN(sel,:), buildRegressionNet(D.numSampled,NB,96), "mse", opt());
    R = evalTopK(predict(net,D.XtestNN), D);
    fprintf('%8.2f | %6d | %4s | %7.1f\n', f, m, num2str(R.summary.K90_Neural), R.accNeural(13));
end
disp('EXP_ABL_BASE_DONE');

function layers = mlp(nin, nout, w, nLayers)
    layers = featureInputLayer(nin, Name="input", Normalization="zscore");
    for i=1:nLayers
        layers = [layers; fullyConnectedLayer(w); leakyReluLayer(0.01)]; %#ok<AGROW>
    end
    layers = [layers; fullyConnectedLayer(nout); tanhLayer];
end
