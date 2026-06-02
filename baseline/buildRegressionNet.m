function layers = buildRegressionNet(numSampled, numBeamPairs, hiddenWidth)
%buildRegressionNet  Layers for the baseline RSRP-regression beam-selection net.
%
%   Reconstructs the canonical MathWorks regression architecture: a 4-hidden
%   layer MLP that maps NUMSAMPLED downsampled RSRP measurements to the full
%   NUMBEAMPAIRS-element RSRP profile. Input is z-score normalised; the output
%   passes through tanh (targets are normalised to [-1,1]).
%
%   See also loadBeamData, run_baseline.

    arguments
        numSampled   (1,1) double
        numBeamPairs (1,1) double
        hiddenWidth  (1,1) double = 96
    end

    layers = [ ...
        featureInputLayer(numSampled, Name="input", Normalization="zscore")

        fullyConnectedLayer(hiddenWidth, Name="linear1")
        leakyReluLayer(0.01, Name="leakyRelu1")

        fullyConnectedLayer(hiddenWidth, Name="linear2")
        leakyReluLayer(0.01, Name="leakyRelu2")

        fullyConnectedLayer(hiddenWidth, Name="linear3")
        leakyReluLayer(0.01, Name="leakyRelu3")

        fullyConnectedLayer(hiddenWidth, Name="linear4")
        leakyReluLayer(0.01, Name="leakyRelu4")

        fullyConnectedLayer(numBeamPairs, Name="linear5")
        tanhLayer(Name="tanh") ];
end
