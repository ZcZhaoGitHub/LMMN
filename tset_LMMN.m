clear all,
close all
clc
%% data
[ inputDimension,np,trainSize,testSize ] = deal( 10 , 10^(-6) , 50000 , 10000 );
% systemIdentify
[ trainInput ,trainTarget, testInput,testTarget,Xnp ] = ...
             systemIdentify( inputDimension,np,trainSize,testSize );
% stepSize and count
[ lr,count ] = deal(linspace(0.01,0.1,10)',10);
% initialization
[ Simulation,Theory1,Theory2,MSE1,MSE2 ] = deal(...
             zeros(count,1),zeros(count,1),zeros(count,1),zeros(trainSize,1),zeros(trainSize,1));
% set
[ delte ] = deal( 0.5 );
[ Xi_4,Xi_6,delte_ ] = deal(mean(Xnp.^4),mean(Xnp.^6),(1-delte));
a = delte*delte*np + 2*delte*delte_*Xi_4 + delte_*delte_*Xi_6;
b = delte + 3*delte_*np;
c = delte*delte + 12*delte*delte_*np + 15*delte_*Xi_4;
%% algorithm
for i = 1:count

    [ MSE ] = LMMN1( trainInput,trainTarget,testInput,testTarget,lr(i),delte );  
    for n = 1:trainSize
    R = trainInput(:,n)*trainInput(:,n)';
    MSE1(n) = lr(i)*a*trace(R)/(2*b)+np;%small u
    MSE2(n) = (lr(i)*a*trace(R))/(2*b-lr(i)*c*trace(R))+np;%large u
    end
    
    Simulation(i) = mean(MSE(trainSize-5000:trainSize));
    Theory1(i) = mean(MSE1(trainSize-5000:trainSize));
    Theory2(i) = mean(MSE2(trainSize-5000:trainSize));
%{
    MSE1(n) = lr(i)*np*trace(R)/2 + np;%small u
    MSE2(n) = (lr(i)*np*trace(R))/(2-lr(i)*trace(R)) + np;%large u
%}
end
%% plot
figure
plot(lr,Simulation,'r:<','LineWidth',2);
hold on
plot(lr,Theory1,'b-o','LineWidth',2);
hold on
plot(lr,Theory2,'g-*','LineWidth',2);
grid on
set(gca,'FontSize',14)
set(gca,'FontName','Arial');
legend('Simulation','Theory(Small \mu)','Theory(Large \mu)')
title(' Experimental and theoretical MSE versus \mu for LMMN');
xlabel('\mu')
ylabel('MSE')