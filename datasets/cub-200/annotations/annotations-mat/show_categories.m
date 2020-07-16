% 
% written by Takeshi Mita - March 2009
%
function show_categories()

base_dir = '/common/Image_Datasets/Birds200';

categories = dir(base_dir);
categories = categories(3:end); % remove . and ..
n_categories = length(categories);
if n_categories ~= 200
    fprintf('Strange number of categories!\n');
    return;
end

figure(1);
i = 1;
for c = 1 : n_categories
    category_name = strrep(categories(c).name, '_', ' ');

    image_files = dir([base_dir '/' categories(c).name '/*.jpg']);
    n_images = length(image_files);

    % load the image
    image_file = [base_dir '/' categories(c).name '/' image_files(1).name];
    ima = imread(image_file);
    
    % load the annotation
    annotation_file = strrep(image_file, '/Birds200/', '/Birds200_Annotations/');
    annotation_file = strrep(annotation_file, '.jpg', '.mat');
    load(annotation_file, 'seg', 'bbox', 'wikipedia_url', 'flickr_url');

    % show the image
    subplot(5,4,i);
    imshow(ima);
    title(category_name);
    axis image;
    axis ij;
    hold on;
    box_handle = rectangle('position', [bbox.left bbox.top bbox.right-bbox.left+1 bbox.bottom-bbox.top+1]);
    set(box_handle, 'edgecolor', 'y', 'linewidth', 3);
    hold off;
    i = i + 1;

    % show the annotation
    subplot(5,4,i);
    imshow(seg);
    title(strrep(wikipedia_url, '_', '\_'));
    axis image;
    axis ij;
    hold on;
    box_handle = rectangle('position', [bbox.left bbox.top bbox.right-bbox.left+1 bbox.bottom-bbox.top+1]);
    set(box_handle, 'edgecolor', 'y', 'linewidth', 3);
    hold off;
    i = i + 1;
    
    clear ima;
    clear seg;
    clear bbox;
    clear wikipedia_url;

    if i == 21
        w = waitforbuttonpress;
        % proceed by a mouse-click
        if w == 0
            i = 1;
            clf;
            continue;
        end
    end
end

close;
