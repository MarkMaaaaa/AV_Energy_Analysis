clc
clear all

% Define the base directory containing all scenarios
baseDir = 'C:\Users\zh200\OneDrive\Desktop\Energy\EnergyControl\data\';
scenarios = {'sythentic'}; %'peachtree', 
subScenarios = { 'scenario2'}; %  'scenario1', 'scenario3',
vehicleTypes = { 'CAV', 'HV'}; %'ACC',
fileNames = {'1.trc', '2.trc'};


for s = 1:length(scenarios)
    for ss = 1:length(subScenarios)
        for vt = 1:length(vehicleTypes)
            for fn = 1:length(fileNames)
                % Construct the file path
                filePath = fullfile(baseDir, scenarios{s}, subScenarios{ss}, vehicleTypes{vt}, fileNames{fn});
                
                % Open the file for reading
                fileID = fopen(filePath, 'r');
                
                % Check if the file was opened successfully
                if fileID == -1
                    error('File cannot be opened');
                end
                
                % read the global time in the fourth line
                % Read lines until the fourth one
                for i = 1:3
                    fgetl(fileID);  % Read and discard lines
                end
                fourthLine = fgetl(fileID);  % Read the fourth line
                
                % Extract the timestamp using regular expressions
                timestampPattern = '(\d{4}/\d{1,2}/\d{1,2} \d{1,2}:\d{2}:\d{2}\.\d+)';
                timeMatch = regexp(fourthLine, timestampPattern, 'match');
                
                % Convert the extracted time to datetime object if found
                if ~isempty(timeMatch)
                    % Specify the format of the datetime string
                    timeFormat = 'yyyy/MM/dd HH:mm:ss.SSS';
                    % Create datetime object
                    dateTimeObj = datetime(timeMatch{1}, 'Format', timeFormat);
                    disp(['Extracted DateTime Object: ', datestr(dateTimeObj)]);
                else
                    disp('No timestamp found in the fourth line.');
                end
                
                % Skip the first 14 lines of the file
                for i = 1:10
                    fgetl(fileID);
                end
                
                % Initialize containers for categorized data
                data07E8 = {};
                data07EC = {};
                
                % Read and process the file line by line
                while ~feof(fileID)
                    line = fgetl(fileID);
                    % Split the line based on whitespace
                    splitLine = strsplit(line);
                    % Remove the first (empty) element if present
                    if isempty(splitLine{1})
                        splitLine(1) = [];
                    end
                    % Check if the split line has at least the expected number of elements
                    if length(splitLine) >= 7
                        canID = splitLine{4};
                        switch canID
                            case '07E8'
                                data07E8{end+1} = splitLine;
                            case '07EC'
                                data07EC{end+1} = splitLine;
                        end
                    end
                end
                
                % if the third field is not ID run the following
                % % % for i = 1:length(data07EC)
                % % %     data07EC{i}(3)=[];
                % % % end
                % % % for i = 1:length(data07E8)
                % % %     data07E8{i}(3)=[];
                % % % end
                % Close the file
                fclose(fileID);
        
                data07E8_44 = {};
                data07E8_10 = {};
                data07E8_0D = {};
                data07EC_01 = {};
                data07EC_0B = {};
                data07EC_0D = {};
                
                % Categorize data07E8 based on the eighth element
                for i = 1:length(data07E8)
                    eighthElement = data07E8{i}{8}; % Adjust index as needed
                    switch eighthElement
                        case '44'
                            data07E8_44{end+1} = data07E8{i};
                        case '10'
                            data07E8_10{end+1} = data07E8{i};
                        case '0D'
                            data07E8_0D{end+1} = data07E8{i};
                    end
                end
                
                % Categorize data07EC based on the ninth element
                for i = 1:length(data07EC)
                    ninthElement = data07EC{i}{9}; % Adjust index as needed
                    switch ninthElement
                        case '01'
                            data07EC_01{end+1} = data07EC{i};
                        case '0B'
                            data07EC_0B{end+1} = data07EC{i};
                        case '0D'
                            data07EC_0D{end+1} = data07EC{i};
                    end
                end
       
                MassAirFlowValues = []; %unit in g/s
                Commanded_Air_Fuel_Ratio = [];
                Battery_SOC = []; 
                Voltage = []; %unit in V
                Current = []; %unit in A
                Time = [];
                Speed = [];

                % Iterate through the data07E8_10 entries to calculate mass air flow
                for i = 1:length(data07E8_10)
                    % Extract the hexadecimal values of A and B from the entry
                    hexA = data07E8_10{i}{9};  % Ninth element for A
                    hexB = data07E8_10{i}{10}; % Tenth element for B
                    
                    % Convert A and B from hex to decimal
                    decA = hex2dec(hexA);
                    decB = hex2dec(hexB);
                    
                    % Calculate the mass air flow value
                    MassAirFlow = (256 * decA + decB) / 100;
                    
                    % Append the calculated value to the massAirFlowValues array
                    Time = [Time; dateTimeObj + seconds(str2double(data07E8_10{i}{2})/1000)];
                    MassAirFlowValues = [MassAirFlowValues; MassAirFlow];
                end
                MassAirFlow_table = timetable(Time, MassAirFlowValues);
                Time = [];
                % for i = 1:length(data07E8_44)
                %     % Extract the hexadecimal values of A and B from the entry
                %     hexA = data07E8_44{i}{9};  % Ninth element for A
                %     hexB = data07E8_44{i}{10}; % Tenth element for B
                % 
                %     % Convert A and B from hex to decimal
                %     decA = hex2dec(hexA);
                %     decB = hex2dec(hexB);
                % 
                %     % Calculate the mass air flow value
                %     ratio = (256 * decA + decB)*2 / 65536;
                % 
                %     % Append the calculated value to the massAirFlowValues array
                %     nextCol = size(Commanded_Air_Fuel_Ratio, 2) + 1;
                %     % Commanded_Air_Fuel_Ratio(1, nextCol) = str2double(data07E8_44{i}{2});
                %     Time(nextCol) = dateTimeObj + seconds(str2double(data07E8_44{i}{2}));
                %     Commanded_Air_Fuel_Ratio(nextCol) = ratio;
                % end
                % Commanded_Air_Fuel_Ratio_table = table(Time,Commanded_Air_Fuel_Ratio);
                % clearvars Time
                
                for i = 1:length(data07E8_0D)
                    % Extract the hexadecimal values of A and B from the entry
                    hexA = data07E8_0D{i}{9};  % Ninth element for A
                    % hexB = data07E8_0D{i}{10}; % Tenth element for B
                    
                    % Convert A and B from hex to decimal
                    decA = hex2dec(hexA);
                    % decB = hex2dec(hexB);
                    
                    % Calculate the mass air flow value
                    % speed = (256 * decA + decB) / 100;
                    speed = decA * 0.621371;
                    
                    % Append the calculated value to the massAirFlowValues array
                    Time = [Time; dateTimeObj + seconds(str2double(data07E8_0D{i}{2})/1000)];
                    Speed = [Speed; speed];
                end
                Speed_table = timetable(Time, Speed);
                Time = [];
                
                for i = 1:length(data07EC_01)
                    % Extract the hexadecimal values of A and B from the entry
                    hexA = data07EC_01{i}{10};  % Ninth element for A
                    hexB = data07EC_01{i}{11}; % Tenth element for B
                    
                    % Convert A and B from hex to decimal
                    decA = hex2dec(hexA);
                    decB = hex2dec(hexB);
                    
                    % Calculate the mass air flow value
                    level = (256 * decA + decB) / 1000;
                    
                    % Append the calculated value to the massAirFlowValues array
                    
                    % Battery_SOC(1, nextCol) = str2double(data07EC_01{i}{2});
                    Time = [Time; dateTimeObj + seconds(str2double(data07EC_01{i}{2})/1000)];
                    Battery_SOC = [Battery_SOC; level];
                end
                Battery_SOC_table = timetable(Time,Battery_SOC);
                Time = [];
                
                for i = 1:length(data07EC_0B)
                    % Extract the hexadecimal values of A and B from the entry
                    hexA = data07EC_0B{i}{10};  % Ninth element for A
                    hexB = data07EC_0B{i}{11}; % Tenth element for B
                    
                    % Convert A and B from hex to decimal
                    decA = hex2dec(hexA);
                    decB = hex2dec(hexB);
                    
                    % Calculate the mass air flow value
                    current = (256 * decA + 2*decB) / 1000;
                    
                    % Append the calculated value to the massAirFlowValues array
                    % Current(1, nextCol) = str2double(data07EC_0B{i}{2});
                    Time = [Time; dateTimeObj + seconds(str2double(data07EC_0B{i}{2})/1000)];
                    Current = [Current; current];
                end
                Current_table = timetable(Time,Current);
                Time = [];
                for i = 1:length(data07EC_0D)
                    % Extract the hexadecimal values of A and B from the entry
                    hexA = data07EC_0D{i}{10};  % Ninth element for A
                    hexB = data07EC_0D{i}{11}; % Tenth element for B
                    
                    % Convert A and B from hex to decimal
                    decA = hex2dec(hexA);
                    decB = hex2dec(hexB);
                    
                    % Calculate the mass air flow value
                    voltage = (256 * decA + decB) / 100;
                    
                    % Append the calculated value to the massAirFlowValues array
                    % Voltage(1, nextCol) = str2double(data07EC_0D{i}{2});
                    Time = [Time; dateTimeObj + seconds(str2double(data07EC_0D{i}{2})/1000)];
                    Voltage = [Voltage; voltage];
                end
                voltage_table = timetable(Time,Voltage);
                

                % start_time = max([data_fix.currentDateTimeLocal(1),Battery_SOC_table.Time(1),MassAirFlow_table.Time(1)]);
                % end_time = min([data_fix.currentDateTimeLocal(end),Battery_SOC_table.Time(end),MassAirFlow_table.Time(end)]);
                
                start_time = max([Battery_SOC_table.Time(1),MassAirFlow_table.Time(1)]);
                end_time = min([Battery_SOC_table.Time(end),MassAirFlow_table.Time(end)]);
                % data_fix(data_fix.currentDateTimeLocal < start_time | data_fix.currentDateTimeLocal > end_time,:) = [];
                Battery_SOC_table(Battery_SOC_table.Time < start_time | Battery_SOC_table.Time > end_time, :) = [];
                MassAirFlow_table(MassAirFlow_table.Time < start_time | MassAirFlow_table.Time > end_time, :) = [];
                Current_table(Current_table.Time < start_time | Current_table.Time > end_time, :) = [];
                voltage_table(voltage_table.Time < start_time | voltage_table.Time > end_time, :) = [];
                Speed_table(Speed_table.Time < start_time | Speed_table.Time > end_time, :) = [];
              
                Battery_SOC_table(find(Battery_SOC_table.Battery_SOC==0),:) = []
                


                %capacity of MKZ is 1.4 kWh = 5040000 J
                %resample them to 10 hz
                newTimeStamps = (Battery_SOC_table.Time(1):milliseconds(10):Battery_SOC_table.Time(end))';
                Battery_SOC_table_resampled = retime(Battery_SOC_table, newTimeStamps,'linear');
                Speed_table_resampled = retime(Speed_table,newTimeStamps,"linear");
                MassAirFlow_table_resampled = retime(MassAirFlow_table,newTimeStamps,"linear");
                

                %calculate energy consmuption every 10ms
                Battery_J = diff(Battery_SOC_table_resampled.Battery_SOC)*5040000*0.5;
                % Calorific value of 87 gasoline is 44000 J/g, assume effeciency of the
                % engin is around 40%, airfule ratio is 14.7
                Engine_J = MassAirFlow_table_resampled.MassAirFlowValues/14.7*44000*0.4;
                Engine_J(1) = [];
                Total_J = Engine_J+Battery_J; 
                
                
                % 去掉原始数据的第一项以对齐时间步长
                Time = Battery_SOC_table_resampled.Time(2:end);
                Battery_SOC = Battery_SOC_table_resampled.Battery_SOC(2:end);
                Speed = Speed_table_resampled.Speed(2:end);
                MassAirFlow = MassAirFlow_table_resampled.MassAirFlowValues(2:end);
                
                
                % 创建一个新的表格，包含时间和所有需要的数据
                final_table = table(Time, ...
                                    Battery_SOC, ...
                                    Speed, ...
                                    MassAirFlow, ...
                                    Battery_J, ...
                                    Engine_J, ...
                                    Total_J, ...
                                    'VariableNames', {'Time', 'Battery_SOC', 'Speed', 'MassAirFlow', 'Battery_J', 'Engine_J', 'Total_J'});

                joinedStr = strjoin({scenarios{s}, subScenarios{ss}, vehicleTypes{vt}, fileNames{fn}}, '-');
                finalStr = erase(joinedStr, joinedStr(end-3:end));
                outputPath = fullfile(baseDir, strcat(finalStr, ".csv"));
                disp(outputPath)
                writetable(final_table, outputPath);

            end
        end
    end
end