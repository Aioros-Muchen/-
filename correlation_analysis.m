power = xlsread('data.xlsx', 'A1:A1728'); % 功率
temperature = xlsread('data.xlsx', 'B1:B1728'); % 温度
dewpoint = xlsread('data.xlsx', 'C1:C1728'); % 露点
atmosphericpressure = xlsread('data.xlsx', 'D1:D1728'); % 气压
winddirection = xlsread('data.xlsx', 'E1:E1728'); % 风向
windspeed = xlsread('data.xlsx', 'F1:F1728'); % 风速
humidity = xlsread('data.xlsx', 'G1:G1728'); % 湿度
radiation = xlsread('data.xlsx', 'H1:H1728'); % 辐射

data=[power, temperature, dewpoint, atmosphericpressure, winddirection, windspeed, humidity, radiation];

rho = corr(data, 'type', 'pearson');
% rho = corr(data, 'type', 'Spearman');

string_name = {'power', 'temperature', 'dewpoint', 'atmosphericpressure', 'winddirection', 'windspeed', 'humidity', 'radiation'};
xvalues = string_name.';
yvalues = string_name;

h = heatmap(xvalues, yvalues, rho, 'FontSize', 10, 'FontName', 'Times New Roman');
H.Title = 'pearson 相关系数矩阵';
colormap gray
