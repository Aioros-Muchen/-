%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

tic

%%  读取数据
data1 = xlsread('data.xlsx', 'A1:A1440'); % 训练集。
data2 = xlsread('data.xlsx', 'A1441:A1728'); % 验证集。
temperature = xlsread('data.xlsx', 'B1:B1727'); % 温度
% dewpoint = xlsread('data.xlsx', 'C1:C1727'); % 露点
% windspeed = xlsread('data.xlsx', 'F1:F1727'); % 风速
% humidity = xlsread('data.xlsx', 'G1:G1727'); % 湿度
radiation = xlsread('data.xlsx', 'H1:H1727'); % 辐射
data = [data1; data2]; % 矩阵拼接。

%%  建立模型
numFeatures = 1; % 特征的维数为XTrain的维数
numResponses = 1; % 输出是一维
numHiddenUnits = 200; % 创建LSTM回归网络，指定LSTM层的隐含单元个数200
% gru初始化
grulayers = [ ...
    sequenceInputLayer(numFeatures) % 输入层
    gruLayer(numHiddenUnits, "OutputMode", "sequence") % GRU层
    fullyConnectedLayer(numResponses) % 全连接层，是输出的维数
    regressionLayer]; % 其计算回归问题的半均方误差模块 。即说明这不是在进行分类问题
% cnn初始化
cnnlayers = layerGraph();                                                 % 建立空白网络结构

tempLayers = [
    sequenceInputLayer([3, 1, 1], "Name", "sequence")            % 建立输入层，输入数据结构为[num_dim, 1, 1]
    sequenceFoldingLayer("Name", "seqfold")];                          % 建立序列折叠层
cnnlayers = addLayers(cnnlayers, tempLayers);                                % 将上述网络结构加入空白结构中

tempLayers = convolution2dLayer([2, 1], 32, "Name", "conv_1");         % 卷积层 卷积核[3, 1] 步长[1, 1] 通道数 32
cnnlayers = addLayers(cnnlayers,tempLayers);                                 % 将上述网络结构加入空白结构中

tempLayers = [
    reluLayer("Name", "relu_1")                                        % 激活层
    convolution2dLayer([2, 1], 64, "Name", "conv_2")                   % 卷积层 卷积核[3, 1] 步长[1, 1] 通道数 64
    reluLayer("Name", "relu_2")];                                      % 激活层
cnnlayers = addLayers(cnnlayers, tempLayers);                                % 将上述网络结构加入空白结构中

tempLayers = [
    globalAveragePooling2dLayer("Name", "gapool")                      % 全局平均池化层
    fullyConnectedLayer(16, "Name", "fc_2")                            % SE注意力机制，通道数的1 / 4
    reluLayer("Name", "relu_3")                                        % 激活层
    fullyConnectedLayer(64, "Name", "fc_3")                            % SE注意力机制，数目和通道数相同
    sigmoidLayer("Name", "sigmoid")];                                  % 激活层
cnnlayers = addLayers(cnnlayers, tempLayers);                                % 将上述网络结构加入空白结构中

tempLayers = multiplicationLayer(2, "Name", "multiplication");         % 点乘的注意力
cnnlayers = addLayers(cnnlayers, tempLayers);                                % 将上述网络结构加入空白结构中

tempLayers = [
    sequenceUnfoldingLayer("Name", "sequnfold")                        % 建立序列反折叠层
    flattenLayer("Name", "flatten")                                    % 网络铺平层
    fullyConnectedLayer(1)                                     % 全连接层
    regressionLayer]; % 其计算回归问题的半均方误差模块 。即说明这不是在进行分类问题。
cnnlayers = addLayers(cnnlayers, tempLayers);                                % 将上述网络结构加入空白结构中

cnnlayers = connectLayers(cnnlayers, "seqfold/out", "conv_1");               % 折叠层输出 连接 卷积层输入;
cnnlayers = connectLayers(cnnlayers, "seqfold/miniBatchSize", "sequnfold/miniBatchSize");
% 折叠层输出 连接 反折叠层输入
cnnlayers = connectLayers(cnnlayers, "conv_1", "relu_1");                    % 卷积层输出 链接 激活层
cnnlayers = connectLayers(cnnlayers, "conv_1", "gapool");                    % 卷积层输出 链接 全局平均池化
cnnlayers = connectLayers(cnnlayers, "relu_2", "multiplication/in2");        % 激活层输出 链接 相乘层
cnnlayers = connectLayers(cnnlayers, "sigmoid", "multiplication/in1");       % 全连接输出 链接 相乘层
cnnlayers = connectLayers(cnnlayers, "multiplication", "sequnfold/in");      % 点乘输出

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

%%  modwt 变换
mod = modwt(data, 'sym4'); % MODWT变换。
mod_size = size(mod, 1);

%%  样本熵
sampEn = 0;
for i = 1: mod_size
    sampEn = sampEn + SampleEntropy(1, 0.1 * std(mod(i, :)), mod(i, :), 1);
end
rate = sampEn / mod_size;

imod = [];
for i = 1: mod_size
    sampEn = SampleEntropy(1, 0.1 * std(mod(i, :)), mod(i, :), 1);

    if sampEn <= rate
        disp(['对第',num2str(i),'个modwt分量建模']) % 输出进度。

        %%  数据处理
        [x, y] = data_process(mod(i, :), 1);
        % 归一化
        [x, mappingx] = mapminmax(x', 0, 1);
        [y, mappingy] = mapminmax(y', 0, 1);
        % 划分数据集
        XTrain = x(1: size(data1, 1));
        XTest = x(size(data1, 1): end);
        YTrain = y(1: size(data1, 1));
        YTest = y(size(data1, 1): end);

        %%  gru 训练模型
        net = trainNetwork(XTrain, YTrain, grulayers, options);
        numTimeStepsTest = size(data2, 1); % 修改步长，可自行输入

        YPred = [];
        for k = 1: numTimeStepsTest % 从第1步开始，这里进行numTimeStepsTest次单步预测
            [net, YPred(:, k)] = predictAndUpdateState(net, XTest(:, k));
        end % predictAndUpdateState函数是一次预测一个值并更新网络状态

        % 反归一化
        imod = double([imod; mapminmax('reverse', YPred, mappingy)]);

    else

        %%  ceemdan 分解
        % ceemdan参数设置
        Nstd = 0.02; % 白噪声标准差倍数。
        NR = 1; % 迭代次数。
        MaxIter = 200; % 最大迭代次数。
        %     imf = emd(mod(k, :)); % EMD变换。
        imf = ceemdan(mod(i, :), Nstd, NR, MaxIter); % CEEMDAN变换。
        imf_size = size(imf, 1);

        %%  样本熵
        sampEn = 0;
        for j = 1: imf_size
            sampEn = sampEn + SampleEntropy(1, 0.1 * std(imf(j, :)), imf(j, :), 1);
        end
        rate = sampEn / imf_size;

        iimf = [];
        imf_trend = [];
        for j = 1: imf_size
            sampEn = SampleEntropy(1, 0.1 * std(imf(j, :)), imf(j, :), 1);

            if sampEn <= rate
                imf_trend = [imf_trend; imf(j, :)];

            else
                disp(['对第',num2str(i),'个modwt分量的第',num2str(j),'个ceemdan分量(imf)建模']) % 输出进度。

                %%  数据处理
                [x, y] = data_process(imf(j, :), 1);

                % 归一化
                [x, mappingx] = mapminmax(x', 0, 1);
                [y, mappingy] = mapminmax(y', 0, 1);
%               x = [x; temperature'; dewpoint'; windspeed'; humidity'; radiation'];
                x = [x; temperature'; radiation'];
                % 划分数据集
                XTrain = x(:, 1: size(data1, 1));
                XTest = x(:, size(data1, 1): end);
                YTrain = y(1: size(data1, 1));
                YTest = y(size(data1, 1): end);
                t = {};
                for j = 1: size(YTrain, 2)
                    t{end + 1} = YTrain(:, j);
                end
                % 数据平铺
                %   将数据平铺成1维数据只是一种处理方式
                %   也可以平铺成2维数据，以及3维数据，需要修改对应模型结构
                %   但是应该始终和输入层数据结构保持一致
                P_train =  double(reshape(XTrain, 3, 1, 1, 1440));
                P_test  =  double(reshape(XTest , 3, 1, 1, 288));
                % 数据格式转换
                for j = 1 : 1440
                    Lp_train{j, 1} = P_train(:, :, 1, j);
                end

                for j = 1 : 288
                    Lp_test{j, 1}  = P_test( :, :, 1, j);
                end

                %%  cnn 训练模型
                net = trainNetwork(Lp_train,t,cnnlayers,options);
                numTimeStepsTest = size(data2, 1); % 修改步长，可自行输入

                YPred = {};
                for j = 1: numTimeStepsTest % 从第1步开始，这里进行numTimeStepsTest次单步预测
                    [net, YPred(:, j)] = predictAndUpdateState(net, Lp_test(j, :));
                end % predictAndUpdateState函数是一次预测一个值并更新网络状态

                % 反归一化
                pre = cell2mat(YPred);
                iimf = double([iimf; mapminmax('reverse', pre, mappingy)]);
            end

        end

        disp(['对第',num2str(i),'个modwt分量的ceemdan趋势分量(imf_trend)建模']) % 输出进度。
        %%  数据处理
        [x, y] = data_process(sum(imf_trend), 1);
        % 归一化
        [x, mappingx] = mapminmax(x', 0, 1);
        [y, mappingy] = mapminmax(y', 0, 1);
        % 划分数据集
        XTrain = x(1: size(data1, 1));
        XTest = x(size(data1, 1): end);
        YTrain = y(1: size(data1, 1));
        YTest = y(size(data1, 1): end);

        %%  gru 训练模型
        net = trainNetwork(XTrain, YTrain, grulayers, options);
        numTimeStepsTest = size(data2, 1); % 修改步长，可自行输入

        YPred = [];
        for j = 1: numTimeStepsTest % 从第1步开始，这里进行numTimeStepsTest次单步预测
            [net, YPred(:, j)] = predictAndUpdateState(net, XTest(:, j));
        end % predictAndUpdateState函数是一次预测一个值并更新网络状态

        % 反归一化
        iimf = double([iimf; mapminmax('reverse', YPred, mappingy)]);
        imod = [imod; sum(iimf)];
    end

end

%%  modwt 反变换
h = [mod, imod]; % 将原始数据分解矩阵与预测值MODWT分解矩阵拼接。
pre_value = imodwt(h, 'sym4'); % 对上一行得到的矩阵进行MODWT反变换，得到重构预测数据。
pre_value = pre_value(:, size(data, 1) + 1: end); % 取预测数据的后30%，作为预测值。

%%  分位数区间预测
ciu = [];
cil = [];
Epsilon = [];
Sigma = [];
Norminv = [];
quantiles = [0.1 0.3 0.5]; % 分位数
averdata = xlsread('data.xlsx', 'P1:P288');

for i = 1: 3

    for j = 1: 288
        q = quantiles(i);
        epsilon = averdata(j) - pre_value(j); % 预测误差
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

    for k = 1: 288

        if (true_value(k) < ciu(288 * (i - 1) + k)) && (true_value(k) > cil(288 * (i - 1) + k))
            in_num = in_num + 1;
        end

    end

    PICP = in_num / k;
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
title('MODWT-CEEMDAN-GRU-CNN-Attention')

toc
disp(['运行时间: ',num2str(toc)]);