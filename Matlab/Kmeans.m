clear all;close all;clc
x=load("data.txt");
figure(1);
plot(x(:,1),x(:,2),'k.');
title('Randomly generated data');
opts = statset('Display','final');
[idx,C]=kmeans(x,2,'Distance','cityblock','Options',opts);

figure(2);
plot(x(idx==1,1),x(idx==1,2),'r.','MarkerSize',12)
hold on
plot(x(idx==2,1),x(idx==2,2),'b.','MarkerSize',12)
plot(C(:,1),C(:,2),'kx','MarkerSize',15,'LineWidth',3)
legend('Cluster 1','Cluster 2','Centroids','Location','NW')
title 'Cluster Assignments and Centroids'
hold off