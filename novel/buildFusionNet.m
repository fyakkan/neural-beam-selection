function net = buildFusionNet(numSampled, numPos, numBeamPairs, opts)
%buildFusionNet  Multi-modal fusion network for beam selection (proposed method).
%
%   net = buildFusionNet(NUMSAMPLED, NUMPOS, NUMBEAMPAIRS) returns an
%   initialised two-branch dlnetwork that fuses BOTH modalities to predict the
%   full RSRP profile:
%       - RSRP branch     : NUMSAMPLED downsampled RSRP measurements -> encoder
%       - Position branch : NUMPOS (3D) UE coordinates              -> encoder
%   The two encodings are concatenated and a shared head regresses the
%   NUMBEAMPAIRS-element RSRP profile (tanh output, targets in [-1,1]).
%
%   This is the headline novelty: the RSRP-only NN baseline and the
%   position-only KNN baseline are each single-modality; this network is the
%   first to FUSE sparse RSRP with UE position. Branch widths are chosen so the
%   shared head matches the baseline's 4x96 trunk, keeping capacity comparable.
%
%   Options (name-value):
%       rsrpWidth (96)  width of the RSRP encoder layers
%       posWidth  (32)  width of the position encoder layers
%       headWidth (96)  width of the shared head layers
%       leak    (0.01)  leaky-ReLU negative slope
%
%   See also loadBeamData, buildRegressionNet, run_novel.

    arguments
        numSampled   (1,1) double
        numPos       (1,1) double
        numBeamPairs (1,1) double
        opts.rsrpWidth (1,1) double = 96
        opts.posWidth  (1,1) double = 32
        opts.headWidth (1,1) double = 96
        opts.leak      (1,1) double = 0.01
    end

    rsrpBranch = [
        featureInputLayer(numSampled, Name="rsrp", Normalization="zscore")
        fullyConnectedLayer(opts.rsrpWidth, Name="r_fc1")
        leakyReluLayer(opts.leak, Name="r_relu1")
        fullyConnectedLayer(opts.rsrpWidth, Name="r_fc2")
        leakyReluLayer(opts.leak, Name="r_relu2") ];

    posBranch = [
        featureInputLayer(numPos, Name="pos", Normalization="zscore")
        fullyConnectedLayer(opts.posWidth, Name="p_fc1")
        leakyReluLayer(opts.leak, Name="p_relu1")
        fullyConnectedLayer(opts.posWidth, Name="p_fc2")
        leakyReluLayer(opts.leak, Name="p_relu2") ];

    head = [
        concatenationLayer(1, 2, Name="concat")
        fullyConnectedLayer(opts.headWidth, Name="h_fc1")
        leakyReluLayer(opts.leak, Name="h_relu1")
        fullyConnectedLayer(opts.headWidth, Name="h_fc2")
        leakyReluLayer(opts.leak, Name="h_relu2")
        fullyConnectedLayer(numBeamPairs, Name="out_fc")
        tanhLayer(Name="tanh") ];

    net = dlnetwork;
    net = addLayers(net, rsrpBranch);
    net = addLayers(net, posBranch);
    net = addLayers(net, head);
    net = connectLayers(net, "r_relu2", "concat/in1");
    net = connectLayers(net, "p_relu2", "concat/in2");
    net = initialize(net);
end
