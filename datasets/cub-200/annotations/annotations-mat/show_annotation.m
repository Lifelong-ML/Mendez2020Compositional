% 
% image_file: string
% annotation_file: string
%
% written by Takeshi Mita - March 2009
%
function show_annotation(image_file, annotation_file)
if nargin < 1
    image_file = '/common/Image_Datasets/Birds200/118.House_Sparrow/House_Sparrow_0001_128704753.jpg';
end
if nargin < 2
    annotation_file = '/common/Image_Datasets/Birds200_Annotations/118.House_Sparrow/House_Sparrow_0001_128704753.mat';
end

% load the annotated data
load(annotation_file, 'seg', 'bbox', 'wikipedia_url', 'flickr_url');

% Display
figure(1);
clf;
subplot(1,2,1);
imshow(image_file);
axis image;
axis ij;
hold on;
box_handle = rectangle('position', [bbox.left bbox.top bbox.right-bbox.left+1 bbox.bottom-bbox.top+1]);
set(box_handle, 'edgecolor', 'y', 'linewidth',5);
title(strrep(wikipedia_url, '_', '\_'));

% show rough segmentation
subplot(1,2,2);
imshow(seg);
axis image;
axis ij;
hold on;
box_handle = rectangle('position', [bbox.left bbox.top bbox.right-bbox.left+1 bbox.bottom-bbox.top+1]);
set(box_handle, 'edgecolor', 'y', 'linewidth',5);
title(strrep(flickr_url, '_', '\_'));
