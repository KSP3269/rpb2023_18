clear all; close all; clc

x1=load("data.txt");
x2=load("data.csv");

x = x1;
figure (1);
plot(x(:,1),x(:,2),'.')
