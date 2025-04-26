%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

tic

%%  读取数据
data1 = xlsread('data.xlsx', 'A1:A1440'); % 训练集。
data2 = xlsread('data.xlsx', 'A1441:A1728'); % 验证集。
data = [data1; data2]; % 矩阵拼接。

%%  建立模型
numFeatures = 1; % 特征的维数为XTrain的维数
numResponses = 1; % 输出是一维
numHiddenUnits = 200; % 创建LSTM回归网络，指定LSTM层的隐含单元个数200
% lstm初始化
lstmlayers = [ ...
    sequenceInputLayer(numFeatures) % 输入层
    lstmLayer(numHiddenUnits, "OutputMode", "sequence") % LSTM层
    fullyConnectedLayer(numResponses) % 全连接层，是输出的维数
    regressionLayer]; % 其计算回归问题的半均方误差模块 。即说明这不是在进行分类问题
% gru初始化
grulayers = [ ...
    sequenceInputLayer(numFeatures) % 输入层
    gruLayer(numHiddenUnits, "OutputMode", "sequence") % GRU层
    fullyConnectedLayer(numResponses) % 全连接层，是输出的维数
    regressionLayer]; % 其计算回归问题的半均方误差模块 。即说明这不是在进行分类问题

%%  参数设置
options = trainingOptions('adam', ... % 指定训练选项，求解器设置为adam， 200轮训练
    'ExecutionEnvironment', 'multi-gpu', ...
    'MaxEpochs', 200, ... % 设置最大迭代次数，可调
    'InitialLearnRate', 0.005, ... % 指定初始学习率 0.005，在 125 轮训练后通过乘以因子 0.2 来降低学习率
    'LearnRateSchedule', 'piecewise', ... % 每当经过一定数量的时期时，学习率就会乘以一个系数
    'LearnRateDropPeriod', 125, ... % 乘法之间的纪元数由" LearnRateDropPeriod"控制，可调
    'LearnRateDropFactor', 0.2, ... % 乘法因子由参" LearnRateDropFactor"控制，可调
    'Verbose', 0); % 如果将其设置为true，则有关训练进度的信息将被打印到命令窗口中。默认值为true
%             'Plots','training-progress'); % 绘制训练过程。

%%  ceemdan 分解
% avmd参数设置
alpha = 1;
tau = 1 / 288;
DC = 0;
init = 1;
tol = exp(-5);

for K = 5: 15
    % 自适应分解个数
    imf = VMD(mod(i, :), alpha, tau, K, DC, init, tol); % VMD变换。
    imf_size = size(imf, 1);
    % 计算中心频率
    signal = imf(imf_size, :);
    fs = 1;  % 采样频率
    N = length(signal);  % 信号长度
    Y = fft(signal); % 对信号进行傅里叶变换
    freqs = abs(Y/N).^2;
    center_freq = sum(freqs .* abs(Y).^2) / sum(abs(Y).^2);
    % 自适应分解个数
    imf_n = VMD(mod(i, :), alpha, tau, K + 1, DC, init, tol); % VMD变换。
    imf_size_n = size(imf_n, 1);
    % 计算中心频率
    signal = imf_n(imf_size_n, :);
    Y = fft(signal); % 对信号进行傅
    freqs = abs(Y/N).^2;
    center_freq_n = sum(freqs .* abs(Y).^2) / sum(abs(Y).^2);

    if (abs(center_freq  - center_freq_n ) <= center_freq / 20) || isnan(center_freq_n)
        break
    end

end

iimf = [];
for i = 1: imf_size

    %%  数据处理
    [x, y] = data_process(imf(i, :), 1);
    % 归一化
    [x, mappingx] = mapminmax(x', 0, 1);
    [y, mappingy] = mapminmax(y', 0, 1);
    % 划分数据集
    XTrain = x(1: size(data1, 1));
    XTest = x(size(data1, 1): end);
    YTrain = y(1: size(data1, 1));
    YTest = y(size(data1, 1): end);

    %%  lstm 训练模型
    net = trainNetwork(XTrain, YTrain, lstmlayers, options);
    numTimeStepsTest = size(data2, 1); % 修改步长，可自行输入

    for j = 1: numTimeStepsTest % 从第1步开始，这里进行numTimeStepsTest次单步预测
        [net, YPred(:, j)] = predictAndUpdateState(net, XTest(:, j));
    end % predictAndUpdateState函数是一次预测一个值并更新网络状态

    % 反归一化
    iimf = double([iimf; mapminmax('reverse', YPred, mappingy)]);
end

pre_value = sum(iimf);

%%  分位数区间预测
ciu = [];
cil = [];
Epsilon = [];
Sigma = [];
Norminv = [];
quantiles = [0.1 0.3 0.5]; % 分位数
everdata = xlsread('data.xlsx', 'N1:N288');

for i = 1: 3

    for j = 1: 288
        q = quantiles(i);
        epsilon = everdata(j) - pre_value(j); % 预测误差
        Epsilon = [Epsilon; epsilon];
        lag = 6; % 滞后期数
        sigma = median(abs(Epsilon(max(1, end - lag + 1): end))) * 1.483; % 置信区间
        Sigma = [Sigma; sigma];
        ub = pre_value(j) + sigma * norminv(1 - q/2, 0, 1); % 置信区间上限
        lb = pre_value(j) - sigma * norminv(1 - q/2, 0, 1); % 置信区间下限
        ciu = [ciu; ub];
        cil = [cil; lb];
    end

    Norminv=[Norminv,norminv(1-q/2,0,1)];
end

%%  结果分析
true_value=xlsread('data.xlsx','A1729:A2016'); % 预测集
true_value=true_value';

disp('误差指标')
% 点估计误差指标
rmse = sqrt(mean((true_value - pre_value) .^ 2));
disp(['根均方差（RMSE）：', num2str(rmse)]) % 对异常值比 MAE更敏感。
mae = mean(abs(true_value - pre_value));
disp(['平均绝对误差（MAE）：', num2str(mae)]) % 与规模有关，因此不能在不同时间序列的预测中使用，因为它具有固有的规模差异。
mape = abs(mean(abs(true_value - pre_value)/ true_value));
disp(['平均相对百分误差（MAPE）：', num2str(mape * 100), '%'])
% 置信区间误差指标
for i = 1: 3
    PINAW = Norminv(i) * mean(Sigma(288* (i - 1) + 1: 288 * i))/max(Sigma(288* (i - 1) + 1: 288 * i));
    disp(['区间狭窄程度（PINAW）：', num2str(PINAW * 100), '%'])
    in_num = 0;

    for j = 1: 288

        if (true_value(j) < ciu(288 * (i - 1) + j)) && (true_value(j) > cil(288 * (i - 1) + j))
            in_num = in_num + 1;
        end

    end

    PICP = in_num / j;
    disp(['区间覆盖率（PICP）：', num2str(PICP * 100), '%'])
    ng = 1;
    PINC = 1 - quantiles(i);

    if PICP >= PINC
        Yy = 0;
        CWC = PINAW * (1 + Yy * exp(-ng * (PICP - PINC)));

    else
        Yy = 1;
        CWC = PINAW * (1 + Yy * exp(-ng * (PICP - PINC)));
    end

    disp(['区间覆盖率和狭窄程度的综合考量（CWC）：', num2str(CWC * 100), '%']) % 基于覆盖宽度的标准。
    ACE = PICP - PINC;
    disp(['区间覆盖率与置信的偏差（ACE）：', num2str(ACE * 100), '%'])
    ul = [];
    ul = [ciu(288 * (i - 1) + 1: 288 * i), cil(288 * (i - 1) + 1: 288 * i)];
    CRPS = crps(ul, true_value');
    disp(['连续排列概率得分（CRPS）：', num2str(CRPS)])
end

%%  绘图
x_axis = 1: 200;
fprintf('\n')
figure

for i = 1: 3
    plot(x_axis, ciu(288 * (i - 1) + 65: 288 * i - 24),'k--', 'linewidth', 1)
    hold on
    plot(x_axis, cil(288 * (i - 1) + 65: 288 * i - 24), 'k--', 'linewidth', 1)
    hold on
end

patch([x_axis fliplr(x_axis)], [ciu(1+64:288-24)' fliplr(cil(1+64:288-24)')], [0.9, 0.8, 0.8], 'edgealpha', '0', 'facealpha', '.5')
patch([x_axis fliplr(x_axis)], [ciu(289+64:576-24)' fliplr(cil(289+64:576-24)')], [0.9, 0.6, 0.6], 'edgealpha', '0', 'facealpha', '.5')
patch([x_axis fliplr(x_axis)], [ciu(577+64:864-24)' fliplr(cil(577+64:864-24)')], [0.9, 0.4, 0.4], 'edgealpha', '0', 'facealpha', '.5')
plot(x_axis, true_value(65: 264), 'b:', 'linewidth', 2)
hold on
plot(x_axis, pre_value(65: 264), 'r-.', 'linewidth', 2)
hold on

legend('', '', '', '', '', '', '90%置信区间', '70%置信区间', '50%置信区间', '实际值', '预测值')
xlabel('预测样本')
ylabel('发电量(kW)')
grid on
title('AVMD-GRU')

toc
disp(['运行时间: ',num2str(toc)]);