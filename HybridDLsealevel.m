
clc
clear all
[~,~,data] = xlsread('C:\Users\ihu\Desktop\Book2.xlsx');
data_mat  = cell2mat(data);
% Generate example input sequences with time step as 1
numSequences = 488; % Number of sequences

% Generate temperature data (assuming it varies between 0 and 100)
temperatureData = (data_mat(:,2))';

% Generate humidity data (assuming it varies between 0 and 1)
sealevelData = (data_mat(:,1))';

% Generate target temperature data (predicting next time step's temperature)
targetsealevelData = (data_mat(:,3))'; % Shifted by one time step

trainRatio = 0.8; % 80% training, 20% testing
numTrain = floor(trainRatio * numSequences);


% Concatenate the input sequences along the feature dimension
X = [temperatureData; sealevelData]; % Concatenate temperature and humidity data
Y = targetsealevelData;



XTrain = X(:, 1:numTrain);
YTrain = Y(1:numTrain);

XTest = X(:, numTrain+1:end);
YTest = Y(numTrain+1:end);


% Display the size of the generated input sequences
size(temperatureData)
size(sealevelData)
size(targetsealevelData)
layernumber=10
filternumber=32




layers = [
    sequenceInputLayer(2) % Two input features (temperature and humidity)
    convolution1dLayer(1,filternumber)
    lstmLayer(layernumber)
    fullyConnectedLayer(1) % Single output for predicting temperature
    regressionLayer];
 
 
      


% Specify training options
options = trainingOptions('adam', ...
    'MaxEpochs',500, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.01, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress' ...
    );


% Train the LSTM network
net = trainNetwork(XTrain, YTrain, layers, options);
YPredicted = predict(net, XTest);

% Calculate mean squared error
mse = mean((YPredicted - YTest).^2);
Rdeep= corr(YPredicted',YTest');
disp(['Mean Squared Error: ' num2str(mse)]);
% Forecast future temperature using the trained LSTM network
numFutureSteps = 309; % Number of future steps to forecast
XForecast = XTest(:, end); % Use the last observation as the initial input for forecasting

futureTemperatureForecast = zeros(1, numFutureSteps);
for i = 1:numFutureSteps
    YForecast = predict(net, XForecast);
    futureTemperatureForecast(i) = YForecast;
    XForecast = [YForecast; YForecast]; % Update the input for the next time step
end

% Display the forecasted temperatures
disp('Forecasted temperatures:');
disp(futureTemperatureForecast);
