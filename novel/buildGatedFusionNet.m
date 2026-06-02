function net = buildGatedFusionNet(numSampled, numPos, numBeamPairs, opts)
%buildGatedFusionNet  Gated residual multi-modal beam selector (proposed method).
%
%   Predicts the full RSRP profile as
%       out = tanh( base(rsrp) + gate(rsrp,pos) .* corr(pos) )
%   where:
%     - base : RSRP branch's primary 70-element RSRP estimate (the RSRP-only path),
%     - corr : a position-derived correction (70 elements),
%     - gate : a per-beam sigmoid gate in (0,1) computed from BOTH branches.
%
%   The residual+gate structure lets the network drive gate->0 when RSRP is
%   reliable (recovering the strong RSRP-only baseline at no clean-data cost) and
%   gate->1 to inject the position prior when RSRP measurements are blocked/lost.
%   This targets robustness to mmWave beam blockage, the regime where single-
%   modality RSRP collapses but UE position remains informative.
%
%   See also buildFusionNet, buildRegressionNet, run_novel.

    arguments
        numSampled   (1,1) double
        numPos       (1,1) double
        numBeamPairs (1,1) double
        opts.rsrpWidth (1,1) double = 96
        opts.posWidth  (1,1) double = 32
        opts.gateWidth (1,1) double = 64
        opts.leak      (1,1) double = 0.01
        opts.useGate   (1,1) logical = true   % false -> plain residual: out=tanh(base+corr)
    end
    L = opts.leak;

    rsrpBranch = [
        featureInputLayer(numSampled, Name="rsrp", Normalization="zscore")
        fullyConnectedLayer(opts.rsrpWidth, Name="r_fc1");  leakyReluLayer(L, Name="r_relu1")
        fullyConnectedLayer(opts.rsrpWidth, Name="r_fc2");  leakyReluLayer(L, Name="r_relu2") ];
    baseHead = fullyConnectedLayer(numBeamPairs, Name="base");          % primary RSRP estimate

    posBranch = [
        featureInputLayer(numPos, Name="pos", Normalization="zscore")
        fullyConnectedLayer(opts.posWidth, Name="p_fc1");  leakyReluLayer(L, Name="p_relu1")
        fullyConnectedLayer(opts.posWidth, Name="p_fc2");  leakyReluLayer(L, Name="p_relu2") ];
    corrHead = fullyConnectedLayer(numBeamPairs, Name="corr");          % position correction

    net = dlnetwork;
    net = addLayers(net, rsrpBranch);
    net = addLayers(net, baseHead);
    net = addLayers(net, posBranch);
    net = addLayers(net, corrHead);
    net = addLayers(net, additionLayer(2, Name="add"));          % base + (gated) correction
    net = addLayers(net, tanhLayer(Name="tanh"));
    net = connectLayers(net, "r_relu2", "base");
    net = connectLayers(net, "p_relu2", "corr");
    net = connectLayers(net, "base",    "add/in1");
    net = connectLayers(net, "add",     "tanh");

    if opts.useGate
        gateBranch = [
            concatenationLayer(1, 2, Name="gcat")
            fullyConnectedLayer(opts.gateWidth, Name="g_fc");  leakyReluLayer(L, Name="g_relu")
            fullyConnectedLayer(numBeamPairs, Name="g_out");   sigmoidLayer(Name="gate") ];
        net = addLayers(net, gateBranch);
        net = addLayers(net, multiplicationLayer(2, Name="gmul"));   % gate .* corr
        net = connectLayers(net, "r_relu2", "gcat/in1");
        net = connectLayers(net, "p_relu2", "gcat/in2");
        net = connectLayers(net, "gate",    "gmul/in1");
        net = connectLayers(net, "corr",    "gmul/in2");
        net = connectLayers(net, "gmul",    "add/in2");
    else
        net = connectLayers(net, "corr",    "add/in2");             % plain residual
    end
    net = initialize(net);
end
