warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

load('trainedHypandProCNNModelforYingQi.mat', 'net');
load('preprocessParams.mat', 'ps_input');

newData = readmatrix('postevaluation2.xlsx');  % 替换为您的 Excel 文件名

% 提取特征数据
newFeatures = newData(:, 1:1463)';  % 大小为 [1189, 样本数]

% 如果有真实标签，可以提取（可选）
if size(newData, 2) >= 1464
    trueLabels = newData(:, 1464)';
    ground_truth = double(trueLabels);% 大小为 [1, 样本数]
end

% 使用训练数据的归一化参数对新数据进行归一化
newFeaturesNorm = mapminmax('apply', newFeatures, ps_input);

% 获取新样本的数量
numNewSamples = size(newFeaturesNorm, 2);

% 重塑数据
newFeaturesReshaped = reshape(newFeaturesNorm, [1463, 1, 1, numNewSamples]);

% 使用模型进行分类，获取预测标签和预测概率
[predictedLabels, scores] = classify(net, newFeaturesReshaped);

% 将预测标签转换为数组
predictedLabelsArray = cellstr(predictedLabels);

sampleLabels = {'1', '2'};
% 获取类别列表
classNames = net.Layers(end).Classes;
scores_class1 = scores(:, 1);
scores_class2 = scores(:, 2);
[FP_class1, TP_class1, ~, AUC_class1] = perfcurve(ground_truth, scores_class1, 1);
[FP_class2, TP_class2, ~, AUC_class2] = perfcurve(ground_truth, scores_class2, 2);
disp(['AUC for Hyp: ', num2str(AUC_class1)]);
disp(['AUC for Pro: ', num2str(AUC_class2)]);
roc_data_class1 = [FP_class1, TP_class1];
roc_data_class2 = [FP_class2, TP_class2];
writematrix(roc_data_class1, 'ROC_Hypallpost1.xlsx', 'Sheet', 1, 'Range', 'A1');
writematrix(roc_data_class2, 'ROC_Proallpost1.xlsx', 'Sheet', 1, 'Range', 'A1');
% 显示预测结果
for i = 1:numNewSamples
    % 获取当前样本的预测分数（概率）
    sampleScores = scores(i, :);

    % 获取最高的预测概率和对应的类别
    [maxScore, idx] = max(sampleScores);
    predictedClass = classNames(idx);

    fprintf('样本 %d 的预测类别是: %s，预测概率: %.2f%%\n', i, char(predictedClass), maxScore * 100);
end

% 创建表格
T = table((1:numNewSamples)', predictedLabelsArray, max(scores, [], 2) * 100, ...
    'VariableNames', {'SampleIndex', 'PredictedLabel', 'PredictionProbability'});

% 保存到 Excel 文件
writetable(T, '1Hyp2Proresults.xlsx');

fprintf('预测结果已保存到 PredictionResults.xlsx\n');

if exist('trueLabels', 'var')
    % 将真实标签转换为分类变量
    trueLabelsCategorical = categorical(trueLabels, [1,2], {'1','2'})';

    % 计算准确率
    accuracy = sum(predictedLabels == trueLabelsCategorical) / numNewSamples * 100;

    fprintf('预测准确率为: %.2f%%\n', accuracy);

    % 生成并显示混淆矩阵
    figure;
    cm = confusionchart(trueLabelsCategorical, predictedLabels);
    cm.Title = '新数据集的混淆矩阵';
    cm.ColumnSummary = 'column-normalized';
    cm.RowSummary = 'row-normalized';
end