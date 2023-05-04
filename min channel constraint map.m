clc;
clear all;
clear;
%%
%%
% % generate minimum channel constraint map


image_dir = 'D:\Low\';
result_dir = 'D:\Results\Low\';
image_files = dir([image_dir, '*.png']);
image_len = length(image_files);
win_size=5;
r = 3;
eps = 1e-7;
t0 = 0.2;
for i=1:image_len
    img_name= [image_dir, image_files(i).name];
    img = im2double(imread(img_name));
    dark_channel = get_dark_channel(img,win_size);
    atmosphere = get_atmosphere(img,dark_channel);
    atmosphere = reshape(atmosphere,[1,1,3]);
    I1 = img./atmosphere;
    I2 = 1 - img;
    I2_name = [result_dir, image_files(i).name(1:end-4),'_I2.png'];
    imwrite(I2,I2_name);
    I3=1-img./atmosphere;
    I3_name = [result_dir, image_files(i).name(1:end-4),'_I3.png'];
    imwrite(I3,I3_name);
% For the normal Image, don't need /atmosphere
%     I3=1-img;
    S=min(I3,[],3); 
    attention_name=[result_dir, image_files(i).name(1:end-4),'_attention.png'];
    imwrite(S,attention_name);
 
   
    %%
 
end