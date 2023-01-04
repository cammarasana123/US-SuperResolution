clear; close all;

I = imread('path/to/image'); %load the low-resolution image

SR = superResolution(I); %apply super-resolution algorithm, i.e., cubic convolution

modelfile = 'path/to/network';
CNN_NET = {importKerasNetwork(modelfile)}; %load the trained CNN network

P = predict(CNN_NET_data,SR); %predict the improved super-resolution image