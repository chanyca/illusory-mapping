classdef Directory
    properties
        userID
        ver
        baseDir
    end
    properties (Dependent)
        homeDir
        helperDir
        dataDir
        plotDir
        polarPlotDir
        barPlotDir
        allDataMat
        hotColorMap
    end
    methods
        function obj = Directory()
            obj.userID = extractBetween(pwd, ['Users',filesep], filesep);
            obj.ver = extractAfter(pwd, ['doubleFlash-matlab', filesep]);
            obj.baseDir = extractBefore(pwd, obj.ver);
        end

        function homeDir = get.homeDir(obj)
            if ispc
                homeDir = char(fullfile(obj.baseDir,obj.ver));
            else
                homeDir = ['/Users/Ailene/Documents/GitHub/rabbit-matlab/' obj.ver '/'];
            end
        end
        
        function helperDir = get.helperDir(obj)
            helperDir = char(fullfile(obj.baseDir, 'helper'));
        end

        function dataDir = get.dataDir(obj)
            dataDir = char(fullfile(obj.homeDir, 'Data', 'mat'));
        end

        function plotDir = get.plotDir(obj)
            plotDir = char(fullfile(obj.homeDir, 'plots'));
            if ~isfolder(plotDir)
                mkdir(plotDir)
            end
        end

        function polarplotDir = get.polarPlotDir(obj)
            polarplotDir = char(fullfile(obj.plotDir, 'polar'));
            if ~isfolder(polarplotDir)
                mkdir(polarplotDir)
            end
        end

        function barPlotDir = get.barPlotDir(obj)
            barPlotDir = char(fullfile(obj.plotDir, 'bar'));
            if ~isfolder(barPlotDir)
                mkdir(barPlotDir)
            end
        end


        function allDataMat = get.allDataMat(obj)
            allDataMat = char(fullfile(obj.dataDir, 'all_data.mat'));
        end

        function hotColorMap = get.hotColorMap(obj)
            hotColorMap = char(fullfile(obj.homeDir, 'Hot_custom.mat'));
        end
    end
end
