%% ------------------------------------------------------------------------
%  Real‑Time Trading Analysis
%  – Cumulative profit plot
%  – Prediction error & market price plot
%  – RMSE, MAE, and Directional Accuracy metrics
%% ------------------------------------------------------------------------
clc; clearvars; close all;

% —————————————————————————
% User settings
% —————————————————————————
file       = 'trade_logs_20250729_173312'; %<-- Use your latest log file name
inputName  = [file, '.jsonl'];
outProfit  = [file, '_cumulative_profit.png'];
outError   = [file, '_prediction_error.png'];

% —————————————————————————
% Open JSONL
% —————————————————————————
fid = fopen(inputName, 'r');
if fid == -1
    error('Cannot open %s', inputName);
end

% —————————————————————————
% Init storage
% —————————————————————————
timestamps       = datetime.empty(0,1);
cumulativeProfit = [];
initialBalance   = NaN;
lastBalance      = NaN;
predTimes        = datetime.empty(0,1);
predictedPrice   = [];
spotPrice        = [];

% —————————————————————————
% Read & classify each line
% —————————————————————————
while ~feof(fid)
    line = fgetl(fid);
    if isempty(line) || ~ischar(line), continue; end
    try
        E = jsondecode(line);
        t = datetime(E.datetime,'InputFormat','yyyy-MM-dd HH:mm:ss');
        switch E.event
            case 'initial_balance'
                initialBalance      = E.balance;
                lastBalance         = initialBalance;
                timestamps(end+1,1) = t;
                cumulativeProfit(end+1,1) = 0;
            case 'post_trade_balance'
                if ~isnan(initialBalance)
                    profit = E.balance - lastBalance;
                    lastBalance = E.balance;
                    timestamps(end+1,1)       = t;
                    cumulativeProfit(end+1,1) = cumulativeProfit(end,1) + profit;
                end
            case 'prediction'
                predTimes(end+1,1)      = t;
                predictedPrice(end+1,1) = E.predicted_price;
                spotPrice(end+1,1)      = E.price;
        end
    catch
        % skip malformed lines
    end
end
fclose(fid);

% —————————————————————————
% Sort chronologically & compute errors
% —————————————————————————
[predTimes, idx] = sort(predTimes);
predictedPrice   = predictedPrice(idx);
spotPrice        = spotPrice(idx);
n = numel(predTimes);
errors          = NaN(n,1);
actualPrice1min = NaN(n,1);

% --- START OF FIX ---
% This logic is now more robust and finds the closest timestamp to T+1 minute
for k = 1:n
    targetTime = predTimes(k) + minutes(1);
    
    % Find the absolute time difference between the target and all other points
    time_diffs = abs(predTimes - targetTime);
    
    % Find the index of the point with the minimum time difference
    [min_diff, futureIndex] = min(time_diffs);
    
    % Check if this closest point is within a tolerance (e.g., 5 seconds)
    % and ensure it's not the same point we started with (k).
    if k ~= futureIndex && min_diff <= seconds(5)
        actualPrice1min(k) = spotPrice(futureIndex);
        errors(k) = 100 * abs(predictedPrice(k) - actualPrice1min(k)) / actualPrice1min(k);
    end
end
% --- END OF FIX ---


% —————————————————————————
% Filter to valid data
% —————————————————————————
valid = ~isnan(actualPrice1min);
predTimes_valid       = predTimes(valid);
predictedPrice_valid  = predictedPrice(valid);
actualPrice1min_valid = actualPrice1min(valid);
spotPrice_valid       = spotPrice(valid);
errors_valid          = errors(valid);

% —————————————————————————
% Figure 1: Cumulative Profit
% —————————————————————————
if ~isempty(timestamps) && ~isempty(cumulativeProfit)
    fig1 = figure('Color','w','Units','pixels','Position',[200 200 1000 450]);
    ax1 = axes(fig1);
    plot(ax1, timestamps, cumulativeProfit, 'LineWidth', 1.5);
    grid(ax1,'on');
    ax1.FontSize  = 14; ax1.LineWidth = 1;
    xlabel(ax1,'Time','FontSize',16);
    ylabel(ax1,'Cumulative Profit (USD)','FontSize',16);
    title('Cumulative Trading Profit Over Time', 'FontSize', 18);
    ax1.Position = [0.08 0.15 0.90 0.78];
    xtickformat(ax1,'HH:mm');
    ax1.XAxis.TickLabelRotation = 15;
    print(fig1, outProfit, '-dpng','-r300');
    fprintf('Saved: • %s\n', outProfit);
else
    fprintf('Warning: No profit data found. Skipping cumulative profit plot.\n');
end

% —————————————————————————
% Figure 2: Prediction Error & Metrics
% —————————————————————————
if ~isempty(predTimes_valid)
    % Calculate Metrics
    predicted_move = sign(predictedPrice_valid - spotPrice_valid);
    actual_move    = sign(actualPrice1min_valid - spotPrice_valid);
    predicted_move(predicted_move == 0) = -2;
    correct_directions = sum(predicted_move == actual_move);
    total_valid_predictions = numel(predicted_move);
    directional_accuracy = 100 * correct_directions / total_valid_predictions;
    diffs = predictedPrice_valid - actualPrice1min_valid;
    rmse = sqrt(mean(diffs.^2));
    mae  = mean(abs(diffs));

    % Create Plot
    fig2 = figure('Color','w','Units','pixels','Position',[200 200 1200 550]);
    ax2 = axes(fig2);
    yyaxis(ax2,'left');
    pE = plot(ax2, predTimes_valid, errors_valid, 'LineWidth',1, 'Marker', '.', 'MarkerSize', 0.1);
    pE.Color               = [1 0.4 0.4];
    ax2.YAxis(1).Color     = [0.8 0 0];
    ax2.YAxis(1).Exponent  = 0;
    ax2.YAxis(1).TickLabelFormat = '%.2f%%';
    ylabel(ax2,'Prediction Error (%)','FontSize',16, 'Color', pE.Color);
    
    yyaxis(ax2,'right');
    pM = plot(ax2, predTimes_valid, actualPrice1min_valid, 'LineWidth',1.5);
    pM.Color               = [0.4 0.6 1];
    ax2.YAxis(2).Color     = [0 0 0.8];
    ax2.YAxis(2).Exponent  = 0;
    ax2.YAxis(2).TickLabelFormat = '$%.2f';
    ylabel(ax2,'Actual Market Price at T+1 Min','FontSize',16, 'Color', pM.Color);
    
    grid(ax2,'on');
    ax2.FontSize  = 14; ax2.LineWidth = 1;
    xlabel(ax2,'Time of Prediction','FontSize',16);
    title('1-Minute Price Prediction vs. Actual Market Price', 'FontSize', 18);
    legend([pE, pM], 'Prediction Error', 'Actual Price', 'Location', 'northwest');
    ax2.Position = [0.08 0.15 0.80 0.78];
    xtickformat(ax2,'HH:mm');
    ax2.XAxis.TickLabelRotation = 15;
    print(fig2, outError, '-dpng','-r300');

    % Print final metrics to console
    fprintf('Saved: • %s\n', outError);
    fprintf('\n--- Performance Metrics ---\n');
    fprintf('Mean Absolute Error (MAE):      $%.4f\n', mae);
    fprintf('Root Mean Square Error (RMSE):  $%.4f\n', rmse);
    fprintf('Directional Accuracy:           %.2f%%\n', directional_accuracy);
    fprintf('---------------------------\n');
else
    fprintf('Warning: No valid 1-minute-ahead data points found in the log file.\n');
    fprintf('         Cannot calculate prediction errors or generate the error plot.\n');
end