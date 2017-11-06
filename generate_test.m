clear; close all;
%% settings
folder = 'Test/Set5/';
scale = 3;

%% generate data
filepaths = dir(fullfile(folder,'*.bmp'));

for i = 1 : length(filepaths)
    
    [path, name, ext] = fileparts(filepaths(i).name);
    image = imread(fullfile(folder, filepaths(i).name));
    image = rgb2ycbcr(image);
    image = im2double(image(:, :, 1));
    
    label = modcrop(image, scale);
    data = imresize(label, 1/scale, 'bicubic');
    
    save([folder name '.mat'], 'data', 'label');
end