%%  
clc;clear;close all
data = xlsread('data.xlsx');

x = reshape(data(:,1),288,7);
y = repmat(1:7,288,1);

f = figure('Color','w','Position',[145.5000  174.0000  560.0000  572.0000]);hold on
for i=1:7
    plot3(y(:,i),1:288,x(:,i),'LineWidth',2)
end

xlim([0.5 7.5])
ylim([0 288])
zlim([0 5200])
xlabel('days')
ylabel('points')
zlabel('Generating capacity (kW)')
box on
grid on

set(gca,'XTick',1:7,'YTick',[0 100 200 288])
set(gca,'YDir','reverse')
set(gca,'LineWidth',1,'FontName','Times New Roman','FontSize',15)
view(3)