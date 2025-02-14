%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%%  导入数据
res = readmatrix('HypandPro.xlsx');

avg_1=mean(res(1:9688,:),1);
avg_2=mean(res(9689:22348,:),1);
results=[avg_1;avg_2];
writematrix(results,'average.xlsx');
disp('average')

%%  CNN ANALYSIS
res = readmatrix('HypandPro.xlsx');
temp = randperm(22350);

P_train = res(temp(1: 17880), 1: 1463)';
T_train = res(temp(1: 17880), 1464)';
M = size(P_train, 2);

P_test = res(temp(17881: end), 1: 1463)';
T_test = res(temp(17881: end), 1464)';
N = size(P_test, 2);

[P_train, ps_input] = mapminmax(P_train, 0, 1);
P_test  = mapminmax('apply', P_test, ps_input);

t_train =  categorical(T_train)';
t_test  =  categorical(T_test )';

p_train =  double(reshape(P_train, 1463, 1, 1, M));
p_test  =  double(reshape(P_test , 1463, 1, 1, N));
save('preprocessParams.mat',"ps_input");
layers = [
 imageInputLayer([1463, 1, 1])                                % 输入层
 convolution2dLayer([3, 1], 64, 'Padding', 'same','Name','ConvolutionnalNN_1')          % 卷积核大小为 2*1 生成16个卷积
 batchNormalizationLayer('Name', 'bn1')                       % 批归一化层
 reluLayer('Name', 'relu1')                                                  % relu 激活层
 maxPooling2dLayer([2, 1], 'Stride', [2, 1])                % 最大池化层 大小为 2*1 步长为 [2, 1]
 convolution2dLayer([3, 1], 128, 'Padding', 'same','Name','ConvolutionnalNN_2')          % 卷积核大小为 2*1 生成32个卷积
 batchNormalizationLayer('Name', 'bn2')                     % 批归一化层
 reluLayer('Name', 'relu2')                                 % relu 激活层
 maxPooling2dLayer([2, 1], 'Stride', [2, 1]) 
 convolution2dLayer([3, 1], 64, 'Padding', 'same','Name','ConvolutionnalNN_3') 
 batchNormalizationLayer                                    % 批归一化层
 reluLayer
 maxPooling2dLayer([2, 1], 'Stride', [2, 1])                % 最大池化层 大小为 2*1 步长为 [2, 1]
 dropoutLayer(0.5, 'Name', 'dropout1')
 convolution2dLayer([3, 1], 32, 'Padding', 'same','Name','ConvolutionnalNN_4') 
 batchNormalizationLayer                                    % 批归一化层
 reluLayer
 fullyConnectedLayer(32)
 dropoutLayer(0.5, 'Name', 'dropout2')
 fullyConnectedLayer(2)
 softmaxLayer                                               % 损失函数层
 classificationLayer];                                      % 分类层

options = trainingOptions('adam', ...      % Adam 梯度下降
    'MaxEpochs', 100, ...                  % 最大训练次数 500
    'InitialLearnRate', 1e-4, ...          % 初始学习率为 0.0001
    'L2Regularization', 1e-4, ...          % L2正则化参数
    'MiniBatchSize', 64, ...
    'LearnRateSchedule', 'piecewise', ...  % 学习率下降
    'LearnRateDropFactor', 0.1, ...        % 学习率下降因子 0.1
    'LearnRateDropPeriod', 60, ...        % 经过450次训练后 学习率为 0.001 * 0.1
    'Shuffle', 'every-epoch', ...          % 每次训练打乱数据集
    'ValidationPatience', Inf, ...         % 关闭验证
    'Plots', 'training-progress', ...      % 画出曲线
    'Verbose', false);

net = trainNetwork(p_train, t_train, layers, options);

t_sim1 = predict(net, p_train); 
t_sim2 = predict(net, p_test ); 

T_sim1 = vec2ind(t_sim1');
T_sim2 = vec2ind(t_sim2');

error1 = sum((T_sim1 == T_train)) / M * 100 ;
error2 = sum((T_sim2 == T_test )) / N * 100 ;

analyzeNetwork(layers)

%[T_train, index_1] = sort(T_train);
%[T_test , index_2] = sort(T_test );

%T_sim1 = T_sim1(index_1);
%T_sim2 = T_sim2(index_2);
ground_truth = double(T_test);
scores_class1 = t_sim2(:, 1);
scores_class2 = t_sim2(:, 2);
[FP_class1, TP_class1, ~, AUC_class1] = perfcurve(ground_truth, scores_class1, 1);
[FP_class2, TP_class2, ~, AUC_class2] = perfcurve(ground_truth, scores_class2, 2);
disp(['AUC for Hyp: ', num2str(AUC_class1)]);
disp(['AUC for Pro: ', num2str(AUC_class2)]);

roc_data_class1 = [FP_class1, TP_class1];
roc_data_class2 = [FP_class2, TP_class2];
%thresholds_custom = 0:0.01:1;  % 自定义的阈值范围，从0到1，每步0.01
%FP_class1 = zeros(length(thresholds_custom), 1);
%TP_class1 = zeros(length(thresholds_custom), 1);
%FP_class2 = zeros(length(thresholds_custom), 1);
%TP_class2 = zeros(length(thresholds_custom), 1);
%for i = 1:length(thresholds_custom)
    %threshold = thresholds_custom(i);
    %predicted_labels_class1 = scores_class1 > threshold;
    %predicted_labels_class2 = scores_class2 > threshold;  
    %TP_class1(i) = sum(T_test' == 1 & predicted_labels_class1)/sum(T_test' == 1);
    %FP_class1(i) = sum(T_test' == 2 & predicted_labels_class1)/sum(T_test' == 2);  
    %TP_class2(i) = sum(T_test' == 2 & predicted_labels_class2)/sum(T_test' == 2);
    %FP_class2(i) = sum(T_test' == 1 & predicted_labels_class2)/sum(T_test' == 1);
%end
%AUC_class1 = trapz(FP_class1, TP_class1);
%AUC_class2 = trapz(FP_class2, TP_class2);
%disp(['AUC for Hyp: ', num2str(AUC_class1)]);
%disp(['AUC for Pro: ', num2str(AUC_class2)]);

writematrix(roc_data_class1, 'ROC_Hypalldata.xlsx', 'Sheet', 1, 'Range', 'A1');
writematrix(roc_data_class2, 'ROC_Proalldata.xlsx', 'Sheet', 1, 'Range', 'A1');

figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
ylabel('预测结果')
string = {'训练集预测结果对比'; ['准确率=' num2str(error1) '%']};
title(string)
xlim([1, M])
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'测试集预测结果对比'; ['准确率=' num2str(error2) '%']};
title(string)
xlim([1, N])
grid

figure
cm = confusionchart(T_train, T_sim1);
cm.Title = 'Confusion Matrix for Train Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
    
figure
cm = confusionchart(T_test, T_sim2);
cm.Title = 'Confusion Matrix for Test Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

% Plot ROC curves
figure;
plot(FP_class1, TP_class1, 'r', 'LineWidth', 2); % 绘制类别1的ROC曲线
hold on;
plot(FP_class2, TP_class2, 'b', 'LineWidth', 2); % 绘制类别2的ROC曲线
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve for Class 1 and Class 2');
legend('Class 1', 'Class 2');
grid on;
% 保存训练好的模型到当前工作目录
save('trainedHypandProCNNModelforYingQi.mat', 'net');
%%
% 加载训练好的模型
% Set a fixed seed for reproducibility
load('trainedHypandProCNNModelforYingQi.mat', 'net');

% 获取网络的层
layers = net.Layers;

% 显示每一层的名称和类型
for i = 1:length(layers)
    fprintf('Layer %d: Name = %s, Type = %s\n', i, layers(i).Name, class(layers(i)));
end

% 将网络转换为 layerGraph
lgraph = layerGraph(layers);
% 移除名为 'classoutput' 的输出层
lgraph = removeLayers(lgraph, 'classoutput');
softmaxLayerIdx = find(arrayfun(@(l) isa(l, 'nnet.cnn.layer.SoftmaxLayer'), layers));
%dropoutLayerIdx = find(arrayfun(@(x) isa(x, 'nnet.cnn.layer.DropoutLayer'), layers));
%if ~isempty(dropoutLayerIdx)
    %dropoutLayerName = layers(dropoutLayerIdx).Name;
    %lgraph = removeLayers(lgraph, dropoutLayerName);
%end
% 检查是否存在 'softmax' 层，若有则移除
if ~isempty(softmaxLayerIdx)
    softmaxLayerName = layers(softmaxLayerIdx).Name;
    lgraph = removeLayers(lgraph, softmaxLayerName);
end

% 将 layerGraph 转换为 dlnetwork
dlnet = dlnetwork(lgraph); 
analyzeNetwork(lgraph); 
% 导入数据
res = readmatrix('frequencynormalizedfinal.xlsx');
sample_data = res(1:2, 1:1463)'; 
sample_labels = categorical([1,2],[1,2],{'1','2'});

% 对数据进行归一化
[P_train, ps_input] = mapminmax(res(1:2, 1:1463)', 0, 1);
sample_data = mapminmax('apply', sample_data, ps_input);

% 重塑数据
M_sample = size(sample_data, 2); % 样本数
sample_data = double(reshape(sample_data, [1463, 1, 1, M_sample]));
classes = categories(sample_labels);
num_samples = M_sample;
featureGradients = zeros(1463, num_samples);

% 计算每个样本的特征梯度
for i = 1:num_samples
    inputSample = sample_data(:, :, :, i);
    trueLabel = sample_labels(i);
    featureGradients(:, i) = computeFeatureGradients(dlnet, inputSample, trueLabel, classes);
end

% 显示特征梯度的尺寸
disp('Size of featureGradients:');
disp(size(featureGradients)); 

feature_indices =(1:1463)';
T = table(feature_indices, 'VariableNames', {'FeatureIndex'});
for i = 1:num_samples
% 添加每个样本的特征梯度到表格
    data = featureGradients(:, i); 
    data(data < 0) = 0;
    %data(data > 0) = 0;
    data = abs(data);
    maxVal = max(data);
    normalizedData = data/maxVal;
    sampleVarName = ['Sample_', num2str(i)];
    %T.(sampleVarName) = abs(data);
    T.(sampleVarName) = normalizedData;
end
% 指定文件名
filename = 'Hypandpronormalizedplus.xlsx';

% 将表格写入 Excel 文件
writetable(T, filename);

% 定义计算梯度的函数
function [loss, gradients] = modelGradients(dlnet, dlX, trueLabel, classes)
    % 前向传播
    dlYPred = forward(dlnet, dlX);

    % 手动添加 softmax 操作
    dlYPred = softmax(dlYPred);

    % 将真实标签转换为 one-hot 编码
    T = onehotencode(trueLabel, 1, 'ClassNames', classes);

    % 将 T 转换为 dlarray
    T = dlarray(T);

    % 调整 T 的尺寸以匹配 dlYPred
    T = reshape(T, size(dlYPred));

    % 计算交叉熵损失
    loss = crossentropy(dlYPred, T);

    % 计算损失相对于输入的梯度
    gradients = dlgradient(loss, dlX);
end

function featureGradients = computeFeatureGradients(dlnet, inputSample, trueLabel, classes)
    % 将输入转换为 dlarray
    dlX = dlarray(inputSample, 'SSC');

    % 启用自动微分，计算输出和梯度
    [loss, gradients] = dlfeval(@modelGradients, dlnet, dlX, trueLabel, classes);

    % 提取梯度并转换为列向量
    featureGradients = extractdata(gradients(:));
end