classdef h38901Scenario < handle
%h38901Scenario TR 38.901 system-level scenario builder
%
%   h38901Scenario properties:
%
%   Scenario           - Deployment scenario ("UMi", "UMa", "RMa")
%                        (default "UMa")
%   CarrierFrequency   - Carrier frequency in Hz (default 6e9)
%   InterSiteDistance  - Intersite distance in meters (default 500)
%   NumCellSites       - Number of cell sites (1...19) (default 19)
%   NumSectors         - Number of sectors per cell site (1, 3) (default 3)
%   NumUEs             - Number of UEs to drop per cell (default 10)
%   ChosenUEs          - UE dropping method (false, true) (default true)
%   Wrapping           - Geographical distance-based wrapping (true, false)
%                        (default true)
%   SpatialConsistency - Spatial consistency ("None", "Static", 
%                        "ProcedureA", "ProcedureB") (default "None")
%   FullBufferTraffic  - Traffic configuration ("DL", "UL", "on")
%                        (default "DL")
%   Seed               - Random number generator (RNG) seed (default 0)
%   IndoorRatio        - Indoor UE ratio (default [])
%
%   h38901Scenario properties (read-only):
%
%   CellSites          - Structure array containing nrGNBs and nrUEs
%   ScenarioExtents    - Location and size of the scenario
%   
%   h38901Scenario object functions:
%
%   h38901Scenario     - Create scenario builder
%   configureSimulator - Create sites, sectors, and UEs for scenario
%   addCellSite        - Add a cell site at a specific position
%   dropUEs            - Drop UEs randomly across the system
%   addUEs             - Add UEs to a specific site and sector
%   createChannelLinks - Create the set of channel links for a scenario
%
%   See also h38901Channel, wirelessNetworkSimulator, nrGNB, nrUE.

%   Copyright 2022-2025 The MathWorks, Inc.

    % =====================================================================
    % public interface

    properties (Access=public)

        % Deployment scenario to determine properties of nrGNBs and nrUEs
        % created in the configureSimulator object function ("UMi", "UMa",
        % "RMa") (default "UMa")
        Scenario = "UMa";

        % Carrier frequency in Hz (default 6e9)
        CarrierFrequency = 6e9;

        % Intersite distance in meters (default 500)
        InterSiteDistance = 500;

        % Number of cell sites (1...19) (default 19)
        NumCellSites = 19;

        % Number of sectorized gNBs per cell site (1,3) (default 3). For
        % NumSectors=1, there is one gNB per cell site with no
        % sectorization and isotropic antenna elements. For NumSectors=3,
        % there are three gNBs per cell site with boresight azimuth angles
        % of 30, 150, and -90 degrees and antenna elements according to TR
        % 38.901 Table 7.3-1
        NumSectors = 3;

        % Number of UEs to drop per cell. If ChosenUEs=false, NumUEs
        % specifies the average number. If ChosenUEs=true, NumUEs specifies
        % the exact number (default 10)
        NumUEs = 10;

        % Set ChosenUEs=true to select a number of "chosen" UEs, as
        % described in Rec. ITU-R M.2101-0 Section 3.4.1. The NumUEs
        % property specifies the number of UEs. If ChosenUEs=false, the
        % NumUEs property specifies the average number of UEs per cell i.e.
        % NumCellSites * 3 * NumUEs are dropped across the system in total
        ChosenUEs = true;

        % Enable wrap around calculations, as defined in Rec. ITU-R
        % M.2101-0 Attachment 2 to Annex 1
        Wrapping = true;

        % Spatial consistency. Set to "None" (or false) to apply no spatial
        % consistency procedure. Set to "Static" (or true) to apply 
        % TR 38.901 Section 7.6.3.1 "Spatial consistency procedure". Set to
        % "ProcedureA" or "ProcedureB" to apply Procedure A or Procedure B
        % from TR 38.901 Section 7.6.3.2 "Spatially-consistent UT/BS 
        % mobility modelling" (default "None")
        SpatialConsistency = "None";

        % Traffic configuration when connecting nrUEs to an nrGNB during
        % scenario building ("DL", "UL", "on") (default "DL")
        FullBufferTraffic = "DL";

        % Random number generator seed (default 0)
        Seed = 0;

        % Indoor UE ratio. A scalar between 0 and 1 giving the probability
        % that a randomly-dropped UE will be indoor. If empty, the indoor
        % UE ratio is determined using TR 38.901 Tables 7.2-1 and 7.2-3
        % (default [])
        IndoorRatio = [];

    end

    properties (SetAccess=private)

        % 1-by-NumCellSites structure array with each element 
        % corresponding to a cell site. Each element has the field:
        %
        % Sectors - 1-by-NumSectors structure array with each element
        %           corresponding to a sector. Each element has the fields:
        %
        %           BS  - The nrGNB node representing the base station
        %           UEs - A 1-by-N array of nrUE nodes, representing the 
        %                 UEs attached to the BS. The number of UEs N 
        %                 depends on the UE dropping method (ChosenUEs)
        CellSites;

        % Location and size of the scenario, a four-element vector of the
        % form [left bottom width height]. The elements are defined as 
        % follows:
        %   left   - The X coordinate of the left edge of the scenario in 
        %            meters
        %   bottom - The Y coordinate of the bottom edge of the scenario in 
        %            meters
        %   width  - The width of the scenario in meters, that is, the 
        %            right edge of the scenario is left + width
        %   height - The height of the scenario in meters, that is, the 
        %            top edge of the scenario is bottom + height
        ScenarioExtents;

    end

    methods (Access=public)

        function scenario = h38901Scenario(varargin)
        % Create scenario builder

            % Set properties from name-value arguments
            setProperties(scenario,varargin{:});

            % Set up path loss configuration
            scenario.thePathLossConfig = nrPathLossConfig(Scenario=scenario.Scenario);

            % Initialize RNG
            scenario.theRandStream = RandStream('mt19937ar','Seed',scenario.Seed);

            % Create site positions
            scenario.theSitePositions = createSitePositions(scenario.InterSiteDistance);

            % Create empty auto-correlation matrices
            scenario.theAutoCorrMatrices = [];
            scenario.theFirstCoord = [];

            % Initialize count of UEs dropped
            scenario.theUECount = 0;

        end

        function configureSimulator(scenario,sls)
        % configureSimulator(SCENARIO,SLS) creates cell sites, sectors, BS
        % nodes, and UE nodes according to the scenario, attaches UEs to
        % BSs (including specifying traffic configuration), and attaches
        % all nodes to the wirelessNetworkSimulator object, SLS

            % For each cell site
            numCellSites = scenario.NumCellSites;
            for i = 1:numCellSites

                % Create the cell site (with sectorized cells)
                addCellSite(scenario,sls,NumTransmitAntennas=1,NumReceiveAntennas=1,DuplexMode='TDD',TransmitPower=txPower(scenario),ReceiveGain=6,CarrierFrequency=scenario.CarrierFrequency,ChannelBandwidth=cbw(scenario),SubcarrierSpacing=scs(scenario));

            end

            % Drop UEs and connect them to the cells
            dropUEs(scenario,sls,NumTransmitAntennas=1,NumReceiveAntennas=1,NoiseFigure=9,ReceiveGain=0);

        end

        function BSs = addCellSite(scenario,sls,varargin)
        % BSs = addCellSite(SCENARIO,SLS) adds a cell site to the system.
        % The position of the site is the next uninitialized site in the
        % system layout. The function creates an nrGNB object for each
        % sector at the same position.
        %
        % BSs = addCellSite(SCENARIO,SLS,Name=Value) specifies additional 
        % name-value arguments described below.
        %
        % Position - A row vector containing three numeric values 
        %            representing the [X, Y, Z] position of the site in
        %            meters. The default is to use the next uninitialized
        %            site in the system layout.
        %
        % In addition, you can specify any nrGNB object property as a
        % name-value argument to initialize the nrGNB objects that the
        % function creates when adding the cell site to the system.

            % Create cell site
            cellSite = createCellSite(scenario,varargin{:});

            % Record cell site here
            recordCellSite(scenario,cellSite);

            % Add the cell site to the wirelessNetworkSimulator
            BSs = cat(2,cellSite.Sectors.BS);
            addNodes(sls,BSs);

        end

        function UEs = dropUEs(scenario,sls,varargin)
        % UEs = dropUEs(SCENARIO,SLS) drops UEs randomly across the system
        % and attaches the UEs to BSs by path loss. The function attaches
        % the UE nodes to the wirelessNetworkSimulator object, SLS.
        %
        % UEs = dropUEs(SCENARIO,SLS,Name=Value) specifies additional
        % name-value arguments described below.
        %
        % TXRUVirtualization   - Structure specifying the parameters for 
        %                        TR 36.897 Section 5.2.2 TXRU 
        %                        virtualization model option-1B. The
        %                        structure has the following fields:
        %                           K    - Vertical weight vector length
        %                           Tilt - Tilting angle in degrees
        %                           L    - Horizontal weight vector length
        %                           Pan  - Panning angle in degrees
        %                        The default value is 
        %                        struct(K=1,Tilt=0,L=1,Pan=0).
        % DropMode             - Specified as 'CouplingLoss' or 'PathLoss'.
        %                        Specifies whether UEs are attached to the
        %                        BS with the maximum coupling loss or path
        %                        loss during UE dropping. For 'PathLoss',
        %                        the LOS angle between the site and the UE
        %                        is used to determine the sector. The
        %                        default value is 'PathLoss'.
        % TransmitAntennaArray - Structure specifying the transmit antenna 
        %                        array characteristics, see 
        %                        nrCDLChannel/TransmitAntennaArray
        %                        for details.
        % ReceiveAntennaArray  - Structure specifying the receive antenna 
        %                        array characteristics, see 
        %                        nrCDLChannel/ReceiveAntennaArray
        %                        for details.
        %
        % In addition, you can specify any nrUE object property as a
        % name-value argument to initialize the dropped UE nodes.

            % Create UEs by dropping UEs randomly across the system and
            % attaching to BSs by coupling loss or path loss
            [UEs,sites,sectors] = createUEs(scenario,varargin{:});

            % Record UEs here
            recordUEs(scenario,UEs,sites,sectors);

            % Connect UEs to their respective BSs and add UEs to the 
            % wirelessNetworkSimulator object
            connectUEs(scenario,UEs,sites,sectors);
            addNodes(sls,UEs);

        end

        function UEs = addUEs(scenario,sls,UEs,varargin)
        % UEs = addUEs(SCENARIO,SLS,UEs,Name=Value) adds UEs to a specific
        % site and sector, connects the UEs to the BS for that sector, and
        % connect the UEs to the wirelessNetworkSimulator object, SLS. The
        % following name-value arguments must be specified:
        % 
        % Site   - A scalar integer (1...OBJ.NumCellSites) specifying the 
        %          site in which to add the UEs, or a vector of integers 
        %          specifying the site in which to add each UE.
        % Sector - A scalar integer (1...OBJ.NumSectors) specifying the 
        %          sector in which to add the UEs, or a vector of integers 
        %          specifying the sector in which to add each UE.

            % Determine site and sector
            opts = parseInputs(struct(),varargin{:});
            site = opts.Site;
            sector = opts.Sector;

            % Record UE here
            if (~isscalar(UEs) && isscalar(site))
                site = repmat(site,size(UEs));
            end
            if (~isscalar(UEs) && isscalar(sector))
                sector = repmat(sector,size(UEs));
            end
            recordUEs(scenario,UEs,site,sector);

            % Connect UE to its BS and add UE to the
            % wirelessNetworkSimulator
            connectUEs(scenario,UEs,site,sector);
            addNodes(sls,UEs);

        end

        function [channels,chinfo] = createChannelLinks(scenario,varargin)
        % [CHANNELS,CHINFO] = createChannelLinks(SCENARIO,Name=Value)
        % creates the set of channel links, CHANNELS, for a scenario. This
        % object function implements scenario-specific aspects (3-D node
        % positions and 2-D indoor distance for UEs). It uses
        % h38901Channel/createChannelLink to create the individual channel
        % links.
        % 
        % CHANNELS is a structure array with each element specifying a
        % BS-UE channel link. The structure has the following fields:
        % CenterFrequency     - The center frequency of the link in Hz.
        % NumTransmitAntennas - The number of transmit antennas at the BS. 
        % NumReceiveAntennas  - The number of receive antennas at the UE.
        % LargeScale          - The large scale part of the channel. If
        %                       EvaluatePathLoss=true (see name-value
        %                       argument below), it is a scalar specifying
        %                       the power gain in dB resulting from path
        %                       loss, O2I penetration loss and shadow
        %                       fading. Note that the value is negative -
        %                       it is described as a gain, but its value
        %                       will always represent a loss. If
        %                       EvaluatePathLoss=false, it is a function
        %                       handle which accepts node positions and
        %                       calculates the path loss for those
        %                       positions.
        % SmallScale          - The small scale part of the channel, an 
        %                       nrCDLChannel if FastFading=true (see
        %                       name-value argument below) or a structure
        %                       if FastFading=false.
        % TXRUVirtualization  - Structure specifying the parameters for
        %                       TR 36.897 Section 5.2.2 TXRU
        %                       virtualization model option-1B (see
        %                       name-value argument below)
        % PathFilters         - Channel path filter impulse responses, a
        %                       matrix of size Np-by-Nh where Np is the 
        %                       number of paths and Nh is the number of 
        %                       impulse response samples.
        % NodSubs             - A 3-element row vector specifying the site,
        %                       sector and UE subscripts for this link. 
        % NodeSiz             - A 3-element row vector specifying the total
        %                       number of sites, sectors and UEs across all
        %                       links.
        %
        % CHINFO is a structure containing the following fields:
        % AttachedUEInfo      - A structure array with detailed link 
        %                       information for each attached BS-UE link.
        % AllUEInfo           - A structure array with detailed link 
        %                       information for every BS-UE link considered
        %                       during attachment. 
        %
        % The following name-value argument must be specified:
        %
        % SampleRate           - The sample rate of the channel, see
        %                        nrCDLChannel/SampleRate for details.  
        %
        % Additional optional name-value arguments are described below.
        %
        % TXRUVirtualization   - Structure specifying the parameters for 
        %                        TR 36.897 Section 5.2.2 TXRU 
        %                        virtualization model option-1B. The
        %                        structure has the following fields:
        %                           K    - Vertical weight vector length
        %                           Tilt - Tilting angle in degrees
        %                           L    - Horizontal weight vector length
        %                           Pan  - Panning angle in degrees
        %                        The default value is 
        %                        struct(K=1,Tilt=0,L=1,Pan=0).
        % DropMode             - Specified as 'CouplingLoss' or 'PathLoss'.
        %                        Specifies whether UEs are attached to the
        %                        BS with the maximum coupling loss or path
        %                        loss during UE dropping. For 'PathLoss',
        %                        the LOS angle between the site and the UE
        %                        is used to determine the sector. The
        %                        default value is 'PathLoss'.
        % TransmitAntennaArray - Structure specifying the transmit antenna 
        %                        array characteristics, see 
        %                        nrCDLChannel/TransmitAntennaArray
        %                        for details. The default is to create the
        %                        array based on the value of the
        %                        NumTransmitAntennas name-value argument
        %                        and the number of sectors.
        % ReceiveAntennaArray  - Structure specifying the receive antenna 
        %                        array characteristics, see 
        %                        nrCDLChannel/ReceiveAntennaArray
        %                        for details. The default is to create the
        %                        array based on the value of the
        %                        NumReceiveAntennas name-value argument.
        % NumTransmitAntennas  - The number of transmit antennas at the BS.
        %                        The default value is 1.
        % NumReceiveAntennas   - The number of receive antennas at the UE.
        %                        The default value is 1.
        % FastFading           - If true, the channel links are created
        %                        with the fast fading model specified in TR
        %                        38.901 Section 7.5 steps 2 - 11. If false,
        %                        only the LOS part of the channel is
        %                        created, sufficient for calculating
        %                        coupling loss. Specifically, steps 4 - 10
        %                        are omitted and a subset of the
        %                        calculations in Step 11 are performed.
        %                        The default value is true.
        % CouplingLossInfo     - If true, the CHINFO structure will include
        %                        the coupling loss for each BS-UE link. If
        %                        false, CHINFO will include the coupling 
        %                        loss only if DropMode='CouplingLoss'. The
        %                        default value is false.
        % EvaluatePathLoss     - If true, the path loss is evaluated for
        %                        the specified node positions and returned
        %                        as a power in dB. If false, the path loss
        %                        is returned as a function handle which
        %                        accepts node positions and calculates the
        %                        path loss for those positions. The default
        %                        value is true.
        % Site                 - Restrict UE dropping to the specified
        %                        site, which must be in the range
        %                        1...NumCellSites. If absent or empty, all
        %                        sites are considered during UE dropping.
        % Sector               - Restrict UE dropping to the specified
        %                        sector, which must be in the range
        %                        1...NumSectors. If absent or empty, all
        %                        sectors are considered during UE dropping.
        % LOSProbability       - A scalar between 0 and 1 giving the 
        %                        probability that a randomly-dropped UE 
        %                        will be in line of sight (LOS) condition. 
        %                        If empty, the LOS probability is 
        %                        determined using TR 38.901 Table 7.4.2-1.
        %                        The default value is [].

            % NOTE: The next steps are from TR 38.901 Section 7.5

            % -------------------------------------------------------------
            % "Step 1 a) Choose one of the scenarios"
            % Given by obj.Scenario

            % -------------------------------------------------------------
            % "Step 1 b) Give number of BS and UT"
            % - number of BS is obj.NumCellSites * obj.NumSectors
            % - number of UT is given in createChannelLinksByLoss

            % -------------------------------------------------------------
            % "Step 1 c) Give 3-D locations of BS and UT"
            % - BS locations are given by the variable 'allsitepos', the 
            %   3-D locations of each site, and the BSs (sectors) within a
            %   site will have the same location
            % - UT locations are given in createChannelLinksByLoss
            h_BS = bsHeight(scenario.Scenario);
            allsitepos = arrayfun(@(x)bsPositions(scenario,x,h_BS),(1:scenario.NumCellSites).','UniformOutput',false);
            allsitepos = cat(1,allsitepos{:});

            % -------------------------------------------------------------
            % Steps 1 d) - g), Steps 2 - 10
            % Create channel links by dropping UEs randomly across the
            % system and attaching to BSs by coupling loss or path loss
            [channels,chinfo] = createChannelLinksByLoss(scenario,allsitepos,varargin{:});
            
            % -------------------------------------------------------------
            % Step 11
            % This step can be performed by executing the SmallScale part
            % of the channels returned by this function

            % -------------------------------------------------------------
            % Step 12
            % Evaluate the large scale channel for each channel link
            opts.EvaluatePathLoss = true;
            opts = parseInputs(opts,varargin{:});
            for i = 1:numel(channels)
                lsfn = channels(i).LargeScale(scenario.thePathLossConfig);
                if (opts.EvaluatePathLoss)
                    ueinfo = chinfo.AttachedUEInfo(i);
                    sitepos = allsitepos(ueinfo.Site,:);
                    uepos = ueinfo.Position;
                    channels(i).LargeScale = -lsfn.execute(sitepos,uepos,scenario.CarrierFrequency);
                else
                    channels(i).LargeScale = @(x,y)-lsfn.execute(x,y,scenario.CarrierFrequency);
                end
            end

        end

    end

    methods

        function v = get.ScenarioExtents(scenario)

            % Ensure that cell sites are initialized
            initializeCellSites(scenario);

            % Get sites
            gNBs = cat(2,cat(2,scenario.CellSites.Sectors).BS);
            if (~isempty(gNBs))
                sites = cat(1,gNBs.Position);
                sites = sites(:,1:2);
            else
                sites = zeros(0,2);
            end

            % Get extents
            ISD = scenario.InterSiteDistance;
            v = getScenarioExtents(ISD,sites);

        end

    end

    % =====================================================================
    % private

    properties (SetAccess=private,Hidden)

        ChannelInfo;

    end

    properties (Access=private)

        theCellSiteCursor;
        theSitePositions;
        thePathLossConfig;
        theRandStream;
        theAutoCorrMatrices;
        theFirstCoord;
        theUECount;

    end

end

%% ========================================================================
%  local functions related to wirelessNetworkSimulator
%  ========================================================================

% Create a cell site
function cellSite = createCellSite(obj,varargin)

    % Ensure that cell sites are initialized
    initializeCellSites(obj);

    % Configure options from name-value arguments
    opts = parseInputs(struct(),varargin{:});

    % Create nrGNBs corresponding to BSs in this site. The nrGNB name
    % stores the intersite distance, site index and sector index, which
    % are used to calculate channel parameters when a channel link for
    % these nodes is requested by wirelessNetworkSimulator
    numCellSites = obj.NumCellSites;
    numSectors = obj.NumSectors;
    h_BS = bsHeight(obj.Scenario);
    h_BS = repmat(h_BS,numSectors,1);
    args = varargin;
    if (isfield(opts,'Position'))
        pos = repmat(opts.Position,numSectors,1);
        posarg = find(cellfun(@(x)isequal(x,'Position'),args));
        args(posarg:posarg+1) = [];
    else
        pos = bsPositions(obj,obj.theCellSiteCursor,h_BS);
    end

    if isfield(opts, 'SubcarrierSpacing')
        subcarrierSpacing = opts.SubcarrierSpacing;
    else
        subcarrierSpacing = scs(obj);
    end

    % Calculate the SRS resource periodicity based on TDD DL-UL
    % configuration if present; otherwise, use the default value (5).
    srsResourcePeriodicity = 5; % (slots)

    if isfield(opts, 'DuplexMode') && strcmp(opts.DuplexMode, 'TDD') && isfield(opts, 'DLULConfigTDD')
        % Validate the DLULConfigTDD
        validateDLULTDDConfig(opts.DLULConfigTDD, subcarrierSpacing);
        srsResourcePeriodicity = calculateSRSResourcePeriodicity(opts.DLULConfigTDD, subcarrierSpacing);
    end

    % Calculate the maximum number of connected UEs
    if (obj.ChosenUEs)
        % With ChosenUEs=true, every site and sector has the same number of
        % UEs, so maximum connected UEs is equal to NumUEs
        maxUE = obj.NumUEs;
    else
        % With ChosenUEs=false, different sites and sectors likely have
        % different numbers of UEs. 'maxUE' is taken to be the same as total
        % number of UEs, because in worst (but extremely unlikely) case all UEs 
        % may be attached to the same site and sector
        maxUE = numCellSites*numSectors*obj.NumUEs;
    end

    % Validate custom SRS transmit periodicity if provided; otherwise, 
    % compute it based on resource periodicity and number of connected UEs
    if isfield(opts, 'SRSPeriodicityUE')
        srsTransmitPeriodicityCustom = opts.SRSPeriodicityUE;
    else
        srsTransmitPeriodicityCustom = [];
    end
    srsTransmissionPeriodicity = calculateSRSTransmissionPeriodicity(maxUE, srsResourcePeriodicity, srsTransmitPeriodicityCustom);

    name = "Wrapping=" + num2str(obj.Wrapping);
    name = name + ",SpatialConsistency=" + num2str(obj.SpatialConsistency);
    name = name + ",ISD=" + sprintf('%0.3f',obj.InterSiteDistance);
    name = name + ",Site=" + obj.theCellSiteCursor + "/" + numCellSites;
    name = name + ",Sector=" + (1:numSectors) + "/" + numSectors;
    nodes = nrGNB(args{:},Position = pos,Name = name,SRSPeriodicityUE=srsTransmissionPeriodicity);
    cellSite = newCellSite(nodes);

end

% Record a cell site in the h38901Scenario object
function recordCellSite(obj,cellSite)

    obj.CellSites(obj.theCellSiteCursor) = cellSite;
    obj.NumCellSites = numel(obj.CellSites);
    obj.theCellSiteCursor = obj.theCellSiteCursor + 1;

end

% Create UEs by dropping UEs randomly across the system and attaching to
% BSs by coupling loss or path loss
function [UEs,sites,sectors] = createUEs(obj,varargin)

    % NOTE: The next steps are from TR 38.901 Section 7.5

    % ---------------------------------------------------------------------
    % "Step 1 a) Choose one of the scenarios"
    % Given by obj.Scenario

    % ---------------------------------------------------------------------
    % "Step 1 b) Give number of BS and UT"
    % - number of BS is obj.NumCellSites * obj.NumSectors
    % - number of UT is given in createChannelLinksByLoss

    % ---------------------------------------------------------------------
    % "Step 1 c) Give 3-D locations of BS and UT"
    % - BS locations are given by the variable 'allsitepos', the 3-D
    %   locations of each site, and the BSs (sectors) within a site
    %   will have the same location
    % - UT locations are given in createChannelLinksByLoss
    cells = cat(1,obj.CellSites.Sectors);
    allsitepos = cat(1,cat(1,cells(:,1).BS).Position);

    % ---------------------------------------------------------------------
    % Steps 1 d) - g), Steps 2, 3, 11 (partial), and 12
    % Create UEs by dropping UEs randomly across the system and attaching
    % to BSs by coupling loss or path loss and record the UE information.
    % To calculate coupling loss, Steps 4 - 10 are omitted and a subset of
    % the calculations in Step 11 are sufficient. Note that the channels
    % produced by createChannelLinksByLoss are thrown away and only the UE
    % positions, numbers of floors, 2-D indoor distances, and attachments
    % to BSs are kept and channels are re-created when the links are active
    % within wirelessNetworkSimulator. Steps 1 d) - g) and Steps 2 - 12 are
    % fully implemented in h38901Channel/channelFunction, which is executed
    % by wirelessNetworkSimulator to apply the channel to a packet for an
    % active link
    [~,chinfo] = createChannelLinksByLoss(obj,allsitepos,varargin{:},SampleRate=1,FastFading=false);
    % ---------------------------------------------------------------------
    
    obj.ChannelInfo = chinfo;

    % Create UE nodes from the positions, numbers of floors, and 2-D indoor
    % distances in the UE information; the node name stores the number of
    % floors and 2-D indoor distance for the UE, which are used to
    % calculate channel parameters when a channel link for this node is
    % requested by wirelessNetworkSimulator. Note that 'totUEs' is the
    % total number of UEs across all the sites and sectors and 'maxUE' is
    % the maximum number of UEs within any site and sector
    ueinfo = chinfo.AttachedUEInfo;
    sites = cat(1,ueinfo.Site);
    sectors = cat(1,ueinfo.Sector);
    totUEs = numel(ueinfo);
    if (obj.ChosenUEs)
        % With ChosenUEs=true, every site and sector has the same number of
        % UEs, so 'ueinfo' is a numCellSites-by-numSectors-by-numUEs array.
        % Therefore 'maxUE' is the size of the 3rd dimension (numUEs)
        maxUE = size(ueinfo,3);
    else
        % With ChosenUEs=false, different sites and sectors likely have
        % different numbers of UEs ('ueinfo' is arranged as a column
        % vector). 'maxUE' is taken to be the same as 'totUEs', because in
        % the worst (but extremely unlikely) case all UEs may be attached
        % to the same site and sector
        maxUE = totUEs;
    end
    pos = zeros(totUEs,3);
    name = strings(totUEs,1);
    for i = 1:totUEs

        BS = cells(sites(i),sectors(i)).BS;
        pos(i,:) = ueinfo(i).Position;
        if (obj.ChosenUEs)
            [~,~,ue] = ind2sub([obj.NumCellSites obj.NumSectors obj.NumUEs],i);
        else
            ue = i;
        end
        name(i) = BS.Name + ",UE=" + ue + "/" + maxUE + ",d_2D_in=" + sprintf('%0.3f',ueinfo(i).d_2D_in) + ",n_fl=" + sprintf('%d',ueinfo(i).n_fl);

    end

    % Create UEs with any name-value arguments that have been provided.
    % Remove name-value arguments that belong to the function here rather
    % than nrUE
    args = varargin;
    for n = ["TXRUVirtualization" "DropMode" "TransmitAntennaArray" "ReceiveAntennaArray" "CouplingLossInfo"]
        argidx = find(cellfun(@(x)isequal(x,n),args));
        args(argidx:argidx+1) = [];
    end
    UEs = nrUE(args{:},Position = pos,Name = name);

end

% Record a set of UEs in the h38901Scenario object
function recordUEs(obj,UEs,sites,sectors)

    for i = 1:numel(UEs)
        obj.CellSites(sites(i)).Sectors(sectors(i)).UEs(end+1) = UEs(i);
    end

end

% Connect a set of UEs to a BS using the nrGNB/connectUE object function,
% including specifying traffic configuration
function connectUEs(obj,UEs,sites,sectors)

    for i = 1:numel(UEs)
        connectUE(obj.CellSites(sites(i)).Sectors(sectors(i)).BS,UEs(i),FullBufferTraffic=obj.FullBufferTraffic);
    end

end

% Initialize proprties that store and access cell sites
function initializeCellSites(obj)

    if (isempty(obj.CellSites))
        % Create cell site array and its cursor
        obj.CellSites = repmat(newCellSite(),1,obj.NumCellSites);
        obj.theCellSiteCursor = 1;
    end

end

% Calculate the SRS resource periodicity based on the TDD DL-UL
% configuration
function srsResourcePeriodicity = calculateSRSResourcePeriodicity(dlULConfigTDD, subcarrierSpacing)

    % minimum SRS resource occurrence periodicity (in slots)
    minSRSResourcePeriodicity = 5;
    numSlotsDLULPattern = dlULConfigTDD.DLULPeriodicity*(subcarrierSpacing/15e3);
    % Set SRS resource periodicity as minimum value such that it is at least 5
    % slots and integer multiple of numSlotsDLULPattern
    allowedSRSPeriodicity = [1 2 4 5 8 10 16 20 32 40 64 80 160 320 640 1280 2560];
    allowedSRSPeriodicity = allowedSRSPeriodicity(allowedSRSPeriodicity>=minSRSResourcePeriodicity & ...
        ~mod(allowedSRSPeriodicity, numSlotsDLULPattern));
    srsResourcePeriodicity = allowedSRSPeriodicity(1);

end

% Calculate the minimum SRS transmission periodicity for the connected UE 
% based on the SRS resource periodicity
function srsTransmissionPeriodicity = calculateSRSTransmissionPeriodicity(numConnectedUEs, srsResourcePeriodicity, srsTransmitPeriodicityCustom)

    % Calculate the minimum SRS transmission periodicity for the connected UEs
    minSRSPeriodicityForGivenUEs = ceil(numConnectedUEs/16)*srsResourcePeriodicity;
    % Calculate the set of SRS transmission periodicity which is a multiple of
    % SRS resource periodicity and valid for the given number of connected UEs
    validSRSPeriodicity = [5 8 10 16 20 32 40 64 80 160 320 640 1280 2560];
    validSet = validSRSPeriodicity(validSRSPeriodicity>=minSRSPeriodicityForGivenUEs & ~mod(validSRSPeriodicity,srsResourcePeriodicity));
    
    if ~isempty(srsTransmitPeriodicityCustom)
        if ismember(srsTransmitPeriodicityCustom, validSet)
            srsTransmissionPeriodicity = srsTransmitPeriodicityCustom;
        else
            % SRS periodicity must be one of the elements in the validSet
            if ~isempty(validSet)
                formattedValidSRSSetStr = [sprintf('{') (sprintf(repmat('%d, ', 1, length(validSet)-1)', validSet(1:end-1) )) sprintf('%d}', validSet(end))];
                messageString = ". Set the SRS periodicity to one of these values: " + formattedValidSRSSetStr + ".";
            else
                messageString = ".";
            end
            error('nr5g:h38901Scenario:InvalidSRSPeriodicityUE','Given SRS transmission periodicity (%d) is either invalid or insufficient for the number of connected UEs (%d)%s', srsTransmitPeriodicityCustom, numConnectedUEs, messageString);
        end
    else
        if ~isempty(validSet)
            srsTransmissionPeriodicity = validSet(1);
        else
            % Maximum number of the connected UEs with the maximum SRS periodicity
            maxUEWithSRSPeriodicity = 16*(validSRSPeriodicity(end)/srsResourcePeriodicity);
            error('nr5g:h38901Scenario:InvalidNumUEs', 'The number of connected UEs must not exceed (%d). Reduce the UEs connected to this gNB.', maxUEWithSRSPeriodicity);
        end
    end

end

% Validate DLULConfigTDD
function validateDLULTDDConfig(dlulConfigTDD, subcarrierSpacing)
 
    validateattributes(dlulConfigTDD, {'struct'}, {'nonempty'}, 'DLULConfigTDD', 'DLULConfigTDD');
    
    if ~isfield(dlulConfigTDD, 'DLULPeriodicity')
        coder.internal.error('nr5g:nrGNB:MissingDLULConfigField', 'DLULPeriodicity');
    end
    
    validSCS = [15e3 30e3 60e3 120e3];
    numerology = find(validSCS==subcarrierSpacing, 1, 'first');
    % Validate the DL-UL pattern duration
    validDLULPeriodicity{1} = { 1 2 5 10 }; % Applicable for scs = 15e3 Hz
    validDLULPeriodicity{2} = { 0.5 1 2 2.5 5 10 }; % Applicable for scs = 30e3 Hz
    validDLULPeriodicity{3} = { 0.5 1 1.25 2 2.5 5 10 }; % Applicable for scs = 60e3 Hz
    validDLULPeriodicity{4} = { 0.5 0.625 1 1.25 2 2.5 5 10 }; % Applicable for scs = 120e3 Hz
    validSet = cell2mat(validDLULPeriodicity{numerology});
    if ~ismember(dlulConfigTDD.DLULPeriodicity, validSet) % DLULPeriodicity is not valid for the specified numerology
        formattedValidSetStr = [sprintf('{') (sprintf(repmat('%.3f, ', 1, length(validSet)-1)', validSet(1:end-1) )) sprintf('%.3f}', validSet(end))];
        coder.internal.error('nr5g:nrGNB:InvalidDLULPeriodicity', ""+dlulConfigTDD.DLULPeriodicity, formattedValidSetStr);
    end

end

%% ========================================================================
%  local functions independent of wirelessNetworkSimulator
%  ========================================================================

% Create channel links by dropping UEs and attaching them to BSs based on
% coupling loss or path loss
function [channels,chinfo] = createChannelLinksByLoss(obj,allsitepos,varargin)

    % Reset state variables inside noFastFadingPathGains by calling it with
    % no input arguments. noFastFadingPathGains is used during coupling
    % loss calculations when fast fading is disabled, as it is during UE
    % attachment. The state variables cached inside noFastFadingPathGains
    % make assumptions about the set of channel links calculated during
    % dropping of UEs: the antenna arrays are the same for all links except
    % that the transmit antenna array orientation differs for each sector
    % in a site, and that all sites have the same sectorization
    h38901Channel.noFastFadingPathGains();

    % Parse name-value argument inputs, with defaults
    opts = struct();
    opts.NumTransmitAntennas = 1;
    opts.NumReceiveAntennas = 1;
    opts.TXRUVirtualization = struct(K=1,Tilt=0,L=1,Pan=0);
    opts.FastFading = true;
    opts.DropMode = 'PathLoss';
    opts.CouplingLossInfo = false;
    opts.Site = [];    
    opts.Sector = [];
    opts.LOSProbability = [];
    % SampleRate is mandatory, so is not defaulted here
    opts = parseInputs(opts,varargin{:});
    pathLossAttach = strcmp(opts.DropMode,'PathLoss');

    % Set up variables for the number of sites, sectors and UEs
    if (isempty(opts.Site))
        numCellSites = obj.NumCellSites;
    else
        numCellSites = 1;
    end
    if (isempty(opts.Sector))
        numSectors = obj.NumSectors;
    else
        numSectors = 1;
    end
    numUEs = obj.NumUEs;

    % ---------------------------------------------------------------------
    % "Step 1 b) Give number of UT"
    % Set up the condition that indicates UE dropping is complete:
    % * ChosenUEs= true: every site and sector has numUEs
    % * ChosenUEs=false: the average number of UEs across the sites and 
    %                    sectors is numUEs
    if (obj.ChosenUEs)
        finfn = @(x)all(x(:)==numUEs);
    else
        finfn = @(x)(sum(x(:))==numCellSites*numSectors*numUEs);
    end
    % ---------------------------------------------------------------------

    % Initialize variables that will store channels and channel information
    ch = h38901Channel.createChannelLink();
    channels = repmat({ch([])},numCellSites,numSectors);
    ueidx = repmat({zeros(0,1)},numCellSites,numSectors);
    newinfo = struct(Config=NaN,Site=NaN,Sector=NaN,Cursor=NaN,Position=[NaN NaN NaN],O2I=NaN,ue_theta=NaN,n_fl=NaN,d_2D_in=NaN,d_2D_out=NaN,pLOS=NaN,LOS=NaN,h_E=NaN,SF=NaN,PL=NaN,CL=NaN,Channel=NaN);
    ueinfo = repmat({newinfo([])},numCellSites,numSectors);
    allueinfo = repmat(newinfo,numCellSites,numSectors,numCellSites*numSectors*numUEs);

    % Get offsets for geographical distance based wrapping, if enabled
    % (Rec. ITU-R M.2101-0 Attachment 2 to Annex 1)
    if (obj.Wrapping)
        offsets = h38901Channel.wrappingOffsets(obj.InterSiteDistance,obj.NumCellSites,obj.NumSectors);
    else
        offsets = [0 0];
    end

    if (numSectors==3 || ~isempty(opts.Sector))
        thetafns = {@(x)(x>-30 & x<=90) @(x)(x>90 | x<=-150) @(x)(x>-150 & x<=-30)};
    else % numSectors==1
        thetafns = {@(x)true};
    end

    % Get the total number of sites, sectors and UEs. This vector is used
    % in conjunction with subscripts for sites, sectors and UEs to control
    % the generation of parameters for the channel links
    siz = [obj.NumCellSites obj.NumSectors size(allueinfo,3)];

    % For the case that this function has been previously called, add the
    % number of UEs previously created to the size vector
    siz(3) = siz(3) + obj.theUECount;

    % For the first call to h38901Channel/createChannelLink, provide the
    % site positions - this will signal that internal state related to
    % autocorrelation of LSPs and other spatially consistent RVs should be
    % reset
    if (obj.theUECount==0)
        sitePositions = allsitepos;
    else
        sitePositions = [];
    end

    % Determine which sites need to be evaluated
    if (isempty(opts.Site))
        slist = 1:numCellSites;
    else
        slist = opts.Site;
    end
    relevantsitepos = allsitepos(slist,:);

    % While UE dropping is not complete
    attached = zeros(numCellSites,numSectors);
    ucursor = 1;
    while (~finfn(attached))

        % Drop an arbitrary number of UEs to attempt to attach in this
        % iteration of the loop.
        numUEsToDrop = 20;

        % -----------------------------------------------------------------
        % "Step 1 c) Give 3-D locations of UT"
        % The UEs are dropped uniformly in the 2-D region that is the union
        % of the sites. Each site is a hexagon as described in TR 38.901,
        % with the InterSiteDistance property specifying the hexagon size.
        % The dropping of a UE consists of choosing its 2-D position, 2-D
        % indoor distance and height. The UEs are dropped such that the 2-D
        % outdoor distance satisfies the minimum distance specified in TR
        % 36.873 Table 6-1.
        if (~spatialConsistency(obj))
            % UE conditions are determined independently of and prior to
            % establishing positions
            [d_2D_in,min_d_2D,n_fl,h_UT] = ueConditions(obj,numUEsToDrop);
            % 2-D UE positions satisfy minimum distance 'min_d_2D' and UE
            % heights in 'h_UT' are assigned
            dropuepos = uePositionsSystemDrop(obj,relevantsitepos,min_d_2D,h_UT);
        else
            % UE positions are established first, because for spatial
            % consistency, the indoor/outdoor state and indoor distance are
            % a spatially-correlated function of the position. 2-D position
            % is not constrained to satisfy a 2-D outdoor minimum distance,
            % and UE height is zero
            min_d_2D = zeros(numUEsToDrop,1);
            h_UT = zeros(numUEsToDrop,1);
            dropuepos = uePositionsSystemDrop(obj,relevantsitepos,min_d_2D,h_UT);
            % UE conditions are determined from 2-D UE positions
            [d_2D_in,min_d_2D,n_fl,h_UT] = ueConditions(obj,relevantsitepos,dropuepos);
            % Assign UE heights in 'h_UT'
            dropuepos(:,3) = h_UT;
            % Remove any UEs that do not satisfy the required 2-D outdoor
            % minimum distance for all sites
            uepos2D = dropuepos(:,1:2);
            sitepos2D = relevantsitepos(:,1:2);
            delta = permute(uepos2D,[1 3 2]) - permute(sitepos2D,[3 1 2]);
            d_2D = vecnorm(delta,2,3);
            r = any(d_2D < min_d_2D,2);
            dropuepos(r,:) = [];
            d_2D_in(r) = [];
            n_fl(r) = [];
        end
        % -----------------------------------------------------------------

        % For each UE
        for u = 1:size(dropuepos,1)

            % Get the UE position
            uepos = dropuepos(u,:);

            % For each site
            maxloss = -Inf;
            for s = slist

                % Get the position of the site
                sidx = find(slist==s);
                sitepos = relevantsitepos(sidx,:);

                % Perform geographical distance based wrapping
                % (Rec. ITU-R M.2101-0 Attachment 2 to Annex 1)
                d = vecnorm((sitepos(1:2) + offsets) - uepos(1:2),2,2);
                [~,idx] = min(d);
                sitepos = sitepos + [offsets(idx,:) 0];

                % Calculate the 2-D distance and azimuth angle between the
                % UE and the site
                v = uepos(1:2) - sitepos(1:2);
                ue_theta = atan2d(v(2),v(1));
                d_2D = vecnorm(v);

                % Calculate the 2-D outdoor distance
                d_2D_out = d_2D - d_2D_in(u);

                if (isempty(opts.Sector))
                    % Determine which sectors need to be evaluated (only
                    % the sector corresponding to the LOS angle if dropping
                    % by PL, all sectors if dropping by CL)
                    if (pathLossAttach)
                        ue_c = find(cellfun(@(x)x(ue_theta),thetafns));
                        clist = ue_c;
                    else
                        clist = 1:numSectors;
                    end
                else
                    % Check if the UE is in the requested sector
                    ue_c = find(cellfun(@(x)x(ue_theta),thetafns));
                    clist = opts.Sector(opts.Sector==ue_c);
                end

                % For each sector
                for c = clist

                    % Create a vector of subscripts for the current site,
                    % sector and UE; this will be used to control the
                    % generation of parameters for the channel link (some
                    % parameters are specific to sites but the same for all
                    % sectors, some are specific to UEs regardless of the
                    % site and sector)
                    subs = [s c obj.theUECount+ucursor];

                    % If the storage for UE information is full, double its
                    % size and update the UE-related element in the vector
                    % which records the size (in the case of
                    % ChosenUEs=true, the total number of UEs attempted for
                    % attachment cannot be determined a priori, so the UE
                    % information must be allowed to grow)
                    if (ucursor > size(allueinfo,3))
                        siz(3) = siz(3) + size(allueinfo,3);
                        allueinfo = cat(3,allueinfo,repmat(newinfo,size(allueinfo)));                        
                    end

                    % -----------------------------------------------------
                    % "Step 1 d) Give BS and UT antenna field patterns F_rx
                    % and F_tx in the global coordinate system and array
                    % geometries"
                    % Determined inside h38901Channel/createChannelLink

                    % -----------------------------------------------------
                    % "Step 1 e) Give BS and UT array orientations with
                    % respect to the global coordinate system"
                    % Determined inside h38901Channel/createChannelLink

                    % -----------------------------------------------------
                    % "Step 1 f) Give speed and direction of motion of UT
                    % in the global coordinate system"
                    % Determined inside h38901Channel/createChannelLink

                    % -----------------------------------------------------
                    % "Step 1 g) Specify system centre frequency f_c and
                    % bandwidth B"
                    % Center frequency is given by obj.CarrierFrequency
                    % Bandwidth is not required for TR 38.901 Section 7.5

                    % -----------------------------------------------------
                    % Steps 2, 3, 11 (partial), and 12
                    % Calling chfn(fastFading) with fastFading=false
                    % performs Steps 2 and 3. Calling 'couplingLoss' on
                    % that channel performs a subset of the calculations in
                    % Step 11 sufficient to calculate coupling loss, and
                    % performs Step 12. Calling 'pathLoss' performs Step
                    % 12.
                    % NOTE: see below for a complete call of Steps 2 - 10 
                    % when fastFading=true

                    % Create channel (which includes a large scale part and
                    % a small scale part)
                    chcfg = struct();
                    chcfg.Seed = obj.Seed;
                    chcfg.Scenario = obj.Scenario;
                    chcfg.InterSiteDistance = obj.InterSiteDistance;
                    chcfg.SitePositions = [];
                    chcfg.HasSmallScale = true;
                    chcfg.FastFading = [];
                    chcfg.NodeSubs = subs;
                    chcfg.NodeSiz = siz;
                    chcfg.SampleRate = opts.SampleRate;
                    chcfg = fieldprecedence(chcfg,opts,'TransmitAntennaArray','NumTransmitAntennas');
                    chcfg = fieldprecedence(chcfg,opts,'ReceiveAntennaArray','NumReceiveAntennas');
                    chcfg.TXRUVirtualization = opts.TXRUVirtualization;
                    chcfg.CenterFrequency = obj.CarrierFrequency;
                    chcfg.BSPosition = sitepos;
                    chcfg.UEPosition = uepos;
                    chcfg.n_fl = n_fl(u);
                    chcfg.d_2D_in = d_2D_in(u);
                    chcfg.Wrapping = obj.Wrapping;
                    chcfg.SpatialConsistency = obj.SpatialConsistency;
                    chcfg.ScenarioExtents = getScenarioExtents(obj.InterSiteDistance,relevantsitepos);
                    chcfg.LOSProbability = opts.LOSProbability;
                    chfn = @(x,y)h38901Channel.createChannelLink(setfield(setfield(chcfg,SitePositions=x),FastFading=y));
                    fastFading = false;
                    [ch,linkinfo] = chfn(sitePositions,fastFading);

                    if (~isempty(sitePositions))
                        % At this point, h38901Channel/createChannelLink
                        % has been called with non-empty sitePositions to
                        % reset internal state related to autocorrelation
                        % of LSPs and other spatially consistent RVs. Now
                        % set sitePositions empty so that state is not
                        % reset on subsequent calls
                        sitePositions = [];
                    end

                    % Calculate coupling loss and/or path loss
                    if (pathLossAttach)
                        PL = pathLoss(obj,ch,sitepos,uepos);
                        CL = NaN;
                        loss = PL;
                    else
                        [CL,PL] = couplingLoss(obj,ch,sitepos,uepos);
                        loss = CL;
                    end
                    % -----------------------------------------------------

                    % Record channel information
                    info.Config = setfield(chcfg,SitePositions=relevantsitepos);
                    info.Site = s;
                    info.Sector = c;
                    info.Cursor = ucursor;
                    info.Position = uepos;
                    info.O2I = linkinfo.O2I;
                    info.ue_theta = ue_theta;
                    info.n_fl = n_fl(u);
                    info.d_2D_in = d_2D_in(u);
                    info.d_2D_out = d_2D_out;
                    info.pLOS = linkinfo.pLOS;
                    info.LOS = linkinfo.LOS;
                    info.h_E = linkinfo.h_E;
                    info.SF = linkinfo.SF;
                    info.PL = PL;
                    info.CL = CL;
                    info.Channel = ch;
                    if (isempty(opts.Sector))
                        cidx = c;
                    else
                        cidx = 1;
                    end
                    allueinfo(sidx,cidx,ucursor) = info;

                    % If this potential attachment has the largest CL or PL
                    % value so far (i.e. the least loss), record it
                    if (loss > maxloss)
                        infomax = info;
                        maxloss = loss;
                        smax = sidx;
                        cmax = cidx;
                        chmax = ch;
                        chfnmax = chfn;
                    end

                end

            end

            % If the UE is not in a requested sector, continue to the next
            % UE
            if (isempty(clist))
                continue;
            end

            % Attach the UE to the BS with the least coupling loss or path
            % loss, unless ChosenUEs=true and this BS has enough UEs
            % already, in which case the UE is thrown away
            if (~obj.ChosenUEs || (attached(smax,cmax) < numUEs))

                % Calculate coupling loss if needed
                if (pathLossAttach && opts.CouplingLossInfo)
                    infomax.CL = couplingLoss(obj,chmax,relevantsitepos(smax,:),infomax.Position);
                    allueinfo(smax,cmax,ucursor).CL = infomax.CL;
                end

                % Create the fast fading channel if needed (the coupling
                % loss calculations above do not need fast fading)
                if (opts.FastFading)
                    % -----------------------------------------------------
                    % Steps 2 - 10
                    % Calling chfn(sitePositions,fastFading) with empty
                    % sitePositions and fastFading=true performs Step 2 -
                    % 10, and the channel returned in the 'channels' output
                    % can later be evaluated to perform Steps 11 and 12
                    chfn = chfnmax;
                    chmax = chfn([],opts.FastFading);
                    % -----------------------------------------------------
                end

                % Record information about the attachment
                attached(smax,cmax) = attached(smax,cmax) + 1;
                channels{smax,cmax}(attached(smax,cmax),1) = chmax;
                ueidx{smax,cmax}(attached(smax,cmax),1) = ucursor;
                ueinfo{smax,cmax}(attached(smax,cmax),1) = infomax;

            end

            % If the UE dropping is complete, break the loop
            if (finfn(attached))
                break;
            end

            % Move to the next UE
            ucursor = ucursor + 1;

        end

    end

    % Prepare final outputs of channels and information, including
    % reshaping the channels and attached UE information into 3-D arrays in
    % the case that ChosenUEs=true
    if (obj.ChosenUEs)
        channels = makeChosenUEArray(channels,numCellSites,numSectors,numUEs);
        ueinfo = makeChosenUEArray(ueinfo,numCellSites,numSectors,numUEs);
    else
        channels = cat(1,channels{:});
        ueinfo = cat(1,ueinfo{:});
    end
    ueidx = cat(1,ueidx{:});
    allueinfo = allueinfo(:,:,ueidx);
    chinfo.AttachedUEInfo = ueinfo;
    chinfo.AllUEInfo = allueinfo;

    % Update the number of UEs created by the object
    obj.theUECount = obj.theUECount + ucursor;

end

% Calculate coupling loss for a channel, including the effect of TXRU
% virtualization
function [CL,PL] = couplingLoss(obj,ch,bspos,uepos)

    PL = pathLoss(obj,ch,bspos,uepos);    
    pathgains = h38901Channel.noFastFadingPathGains(ch);
    pathgains = pathgains * db2mag(PL);
    ch.SmallScale.TransmitAndReceiveSwapped = false;
    pathgains = h38901Channel.applyBeamforming(ch.SmallScale,pathgains,ch.TXRUVirtualization);
    CL = mag2db(sqrt(sum(abs(pathgains).^2,'all')));

end

% Calculate path loss for a channel
function PL = pathLoss(obj,ch,bspos,uepos)

    lsfn = ch.LargeScale(obj.thePathLossConfig);
    PL = -lsfn.execute(bspos,uepos,obj.CarrierFrequency);

end

% BS height according to TR 38.901 Tables 7.2-1 and 7.2-3
function h_BS = bsHeight(s)
    
    h_BS = scenarioSwitch(s,10,25,35);
    
end

% BS position for a site
function pos = bsPositions(obj,siteIndex,h_BS)

    numCellsPerSite = numel(h_BS);
    sitepos = obj.theSitePositions(siteIndex,:);
    pos = repmat(sitepos,numCellsPerSite,1);
    pos(:,3) = h_BS;

end

% UE conditions that constrain UE position: 2-D indoor distance, minimum
% 2-D distance that satisfies the minimum 2-D outdoor distance, and UE
% height
function [d_2D_in,min_d_2D,n_fl,h_UT] = ueConditions(obj,varargin)

    % TR 38.901 Section 7.6.3.3 for indoor state and indoor distance
    s = obj.Scenario;
    rs = obj.theRandStream;
    if (spatialConsistency(obj) && isempty(obj.theAutoCorrMatrices))
        allsitepos = varargin{1};
        ISD = obj.InterSiteDistance;
        extents = getScenarioExtents(ISD,allsitepos);
        minpos = extents(1:2);
        maxpos = minpos + extents(3:4);
        % TR 38.901 Table 7.6.3.1-2
        indoorStateDelta = scenarioSwitch(s,50,50,50);
        % TR 38.901 Section 7.6.3.3
        indoorDistanceDelta = 25;
        % Create auto-correlation matrices, one for indoor state and one
        % for each of the "two spatially consistent uniform random
        % variables" used for indoor distance, as described in Section
        % 7.6.3.3
        distances = [indoorStateDelta repmat(indoorDistanceDelta,1,2)];
        [obj.theAutoCorrMatrices,obj.theFirstCoord] = h38901Channel.createAutoCorrMatrices(rs,minpos,maxpos,distances);
    end

    if (isempty(obj.IndoorRatio))
        % TR 38.901 Tables 7.2-1 and 7.2-3
        indoorRatio = scenarioSwitch(s,0.8,0.8,0.5);
    else
        % User-specified indoor ratio
        indoorRatio = obj.IndoorRatio;
    end
    if (spatialConsistency(obj))
        % Sample indoor state auto-correlation matrix, one sample per UE
        dropuepos = varargin{2};
        numUEs = size(dropuepos,1);
        rvs = h38901Channel.uniformAutoCorrRVs(obj.theAutoCorrMatrices(:,:,1),obj.theFirstCoord,dropuepos(:,1:2));
    else
        numUEs = varargin{1};
        rvs = rs.rand(numUEs,1);
    end
    indoor = rvs < indoorRatio;

    % TR 38.901 Section 7.4.3.1
    d_2D_in_max = scenarioSwitch(s,25,25,10);
    d_2D_in = zeros(numUEs,1);
    if (obj.CarrierFrequency < 6e9)
        % TR 38.901 Table 7.4.3-3, one random variable
        nrv = 1;
    else
        % two random variables (of which the minimum is taken)
        nrv = 2;
    end
    if (spatialConsistency(obj))
        % Sample indoor distance auto-correlation matrices, one sample for 
        % each indoor UE, from one or two matrices depending on 'nrv'
        rvs = zeros(nnz(indoor),nrv);
        for i = 1:nrv
            rvs(:,i) = h38901Channel.uniformAutoCorrRVs(obj.theAutoCorrMatrices(:,:,i+1),obj.theFirstCoord,dropuepos(indoor,1:2));
        end
    else
        rvs = rs.rand(nnz(indoor),nrv);
    end
    d_2D_in(indoor) = min(rvs * d_2D_in_max,[],2);

    % TR 38.901 Tables 7.2-1 and 7.2-3
    min_d_2D_out = scenarioSwitch(obj.Scenario,10,35,35);

    % Error out if intersite distance (and therefore site size) is too
    % small to always satisfy required minimum distance
    ISD = obj.InterSiteDistance;
    if (min_d_2D_out + d_2D_in_max > ISD/sqrt(3))
        error('nr5g:h38901Scenario:InterSiteDistanceTooSmall','Site radius (InterSiteDistance/sqrt(3) = %0.3f m) is smaller than minimum outdoor distance (%0.3f m) + maximum indoor distance (%0.3f m), so UEs cannot be dropped with required minimum distance. Increase InterSiteDistance.',ISD/sqrt(3),min_d_2D_out,d_2D_in_max);
    end

    % Minimum 2-D distance for each UE is minimum 2-D outdoor distance for
    % the scenario plus the 2-D indoor distance for that UE. See TR 36.873
    % Table 6-1 NOTE 1 where "Min. UE-eNB 2D distance" is defined as d_2D
    % for outdoor UEs and d_2D_out for indoor UEs. This implementation
    % follows TR 36.873 Table 6-1 NOTE 1 because it implements the
    % calibration parameters in TR 38.901 Tables 7.8-1 and 7.8-2 where "UT
    % distribution" says "Following TR 36.873"
    min_d_2D_out = min_d_2D_out * ones(numUEs,1);
    min_d_2D = min_d_2D_out + d_2D_in;

    % TR 38.901 Table 7.2-3, 1.5 m height is equivalent to one floor
    n_fl_RMa = ones(numUEs,1);
    % TR 36.873 Table 6-1
    n_fl_UMx = @()ueFloorsIndoorOutdoor(rs,indoor,numUEs);
    n_fl = scenarioSwitch(s,n_fl_UMx,n_fl_UMx,n_fl_RMa);
    h_UT = 3*(n_fl - 1) + 1.5;

end

% Drop UEs randomly across the system (the 2-D region that is the union of
% the sites), subject to the required minimum distance between each UE and
% every BS
function pos = uePositionsSystemDrop(obj,allsitepos,min_d_2D,h_UT)

    % Get polygons that are the boundaries for each site
    [sitex,sitey] = h38901Channel.sitePolygon(obj.InterSiteDistance);
    sysx = allsitepos(:,1) + sitex;
    sysy = allsitepos(:,2) + sitey;
    sysx = sysx(:);
    sysy = sysy(:);

    % Create a predicate that checks if a UE 2-D position (x,y) is inside
    % at least one of the site polygons
    pred = @(x,y)any(arrayfun(@(a,b)inpolygon(x,y,sitex + a,sitey + b),allsitepos(:,1),allsitepos(:,2)));

    % Drop UEs randomly with the x-axis / y-axis bounding box of the union
    % of the site polygons, and subject to the predicate above, and such
    % that the minimum 2-D distance for a given UE is satisfied between
    % that UE and all sites
    pos = uePositionsDrop(obj.theRandStream,sysx,sysy,pred,allsitepos,min_d_2D,h_UT);

end

% TR 36.873 Table 6-1
function n_fl = ueFloorsIndoorOutdoor(rs,indoor,numUEs)

    n_fl = zeros(numUEs,1);
    n_fl(~indoor) = 1;
    N_fl = rs.randi([4 8],sum(indoor),1);
    n_fl(indoor) = arrayfun(@(x)rs.randi(x,[1 1]),N_fl);    

end

% Drop UEs randomly within a given bounding box, subject to a given
% predicate, and with a minimum 2-D distance
function pos = uePositionsDrop(rs,xx,yy,pred,allsitepos,min_d_2D,h_UT)

    % Get x-axis / y-axis bounding box corners
    minx = min(xx);
    maxx = max(xx);
    miny = min(yy);
    maxy = max(yy);

    % While not all UEs have been dropped
    i = 1;
    numUEs = numel(min_d_2D);
    pos = zeros(numUEs,3);
    while (i <= numUEs)

        % For the current UE, randomly select a 2-D position inside the
        % bounding box, and assign the UE height
        x = minx + (maxx - minx)*rs.rand;
        y = miny + (maxy - miny)*rs.rand;
        z = h_UT(i);

        % Calculate the 2-D distances between the UE and all sites
        d = vecnorm([x y] - allsitepos(:,1:2),2,2);

        % If the minimum 2-D distance to all sites and the predicate
        % are satisfied
        if (all(d>=min_d_2D(i)) && pred(x,y))

            % Record this UE position and move on to the next UE
            pos(i,:) = [x y z];
            i = i + 1;

        end

    end

end

% Create a new cell
function c = newCell(varargin)

    c = struct();
    if (nargin==0)
        c.BS = [];
    else
        bs = varargin{1};
        c.BS = bs;
    end
    c.UEs = nrUE.empty;

end

% Create a new cell site
function cs = newCellSite(varargin)

    cs = struct();
    if (nargin==0)
        c = newCell();
        cs.Sectors = c([]);
    else
        cs.Sectors = arrayfun(@newCell,varargin{1});
    end

end

% Set object properties from name-value arguments
function setProperties(obj,varargin)

    s = parseInputs(struct(),varargin{:});
    ns = string(fieldnames(s)).';
    for n = ns
        if (isprop(obj,n))
            obj.(n) = s.(n);
        end
    end

end

% Set structure fields from name-value arguments
function s = parseInputs(s,varargin)

    for i = 1:2:numel(varargin)
        n = varargin{i};
        v = varargin{i+1};
        s.(n) = v;
    end

end

% Configure either field 'f1' or 'f2' of 'chcfg' from 'opts', with field
% 'f1' taking precedence
function chcfg = fieldprecedence(chcfg,opts,f1,f2)

    if (isfield(opts,f1))
        chcfg.(f1) = opts.(f1);
    else
        chcfg.(f2) = opts.(f2);
    end

end

% Calculate the site positions for a given intersite distance, for the
% 19-site layout described in Rec. ITU-R M.2101-0 Section 3.1.1. The sites
% themselves are hexagons as depicted in TR 38.901 Table 7.8-1 (rather than
% smaller per-sector hexagons as depicted in Rec. ITU-R M.2101-0 Figure 2)
function p = createSitePositions(ISD)

    r = ISD * ones(6,1) .* [1 sqrt(3) 2];
    theta = deg2rad((0:60:300).' + [30 0 30]);
    muxfn = @(x)[x(:,1) reshape(x(:,2:3).',[],2)];
    r = muxfn(r);
    theta = muxfn(theta);
    [x,y] = pol2cart([0; theta(:)],[0; r(:)]);
    p = [x y zeros(size(x))];

end

% Set the channel bandwidth according to TR 38.901 Tables 7.8-1 and 7.8-2
function b = cbw(obj)
    
    if (isFR2(obj))
        b = 100e6;
    else
        b = 20e6;
    end

end

% Set the subcarrier spacing to valid values for the channel bandwidths
% specified in TR 38.901 Tables 7.8-1 and 7.8-2
function s = scs(obj)

    if (isFR2(obj))
        s = 60e3;
    else
        s = 15e3;
    end

end

% Determine if a configuration has an FR2 carrier frequency, according to 
% TS 38.104 Table 5.1-1
function fr2 = isFR2(obj)

    fr2 = (obj.CarrierFrequency > 7.125e9);

end

% Set the transmit power according to the scenario
function p = txPower(obj)
    
    % TR 38.901 Tables 7.8-1 and 7.8-2
    if (isFR2(obj))
        pUMa = 35;
        pUMi = 35;
    else        
        pUMa = 49;
        pUMi = 44;
    end
    % TR 38.802 Table A.2.1-1
    pRMa = 49;

    p = scenarioSwitch(obj.Scenario,pUMi,pUMa,pRMa);

end

% Select a value or call a function based on the scenario
function x = scenarioSwitch(s,umi,uma,rma)

    s = lower(string(s));
    if (s=="umi")
        x = select(umi);
    elseif (s=="uma")
        x = select(uma);
    elseif (s=="rma")
        x = select(rma);
    end

    function x = select(x)
        if (isa(x,'function_handle'))
            x = x();
        end
    end

end

% Make an I-by-J-by-K array from an I-by-J cell array with each cell
% containing a vector of K elements
function y = makeChosenUEArray(x,I,J,K)

    y = repmat(x{1,1}(1),[I J K]);
    for i = 1:I
        for j = 1:J
            y(i,j,:) = x{i,j};
        end
    end

end

function v = getScenarioExtents(ISD,sites)

    % Get polygons that are the boundaries for each site
    [sitex,sitey] = h38901Channel.sitePolygon(ISD);
    % Get bounding box of the union of the site polygons
    sysx = sites(:,1) + sitex;
    sysy = sites(:,2) + sitey;
    minpos = [min(sysx, [], 'all'), min(sysy, [], 'all')];
    maxpos = [max(sysx, [], 'all'), max(sysy, [], 'all')];
    v = [minpos maxpos-minpos];

end

function y = spatialConsistency(s)

    if (isnumeric(s.SpatialConsistency) || islogical(s.SpatialConsistency))
        y = s.SpatialConsistency;
    else
        y = ~strcmpi(s.SpatialConsistency,"None");
    end

end
