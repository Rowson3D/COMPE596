close all
clear all
for n = -1:1:5,
figure(n+2);
str = ['T',num2str(10^(n+1))];
eval(str);
h = 0.01;
[X,Y] = meshgrid(0:h:1.0+h,0:h:1.0+h);
surf(X,Y,T);
set(gcf,'color','w')
title(['iterations: ',num2str(10^(n+1))]);
xlabel('x')
ylabel('y')
axis([0 1+h 0 1+h 0 1]);
rotate3d
saveas(gcf,['T',num2str(n+1),'.png'])
end