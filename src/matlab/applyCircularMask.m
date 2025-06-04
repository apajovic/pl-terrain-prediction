function [masked_img] = applyCircularMask(img)
% Primeni kružnu masku na sliku
%
% Ulaz:
%   img - Ulazna slika (matrica, grayscale)
%
% Izlaz:
%   masked_img - Slika sa NaN pikselima van kruga

    % Veličina slike
    [height, width] = size(img);

    % Centar slike
    center = [round(height / 2), round(width / 2)];

    % Maksimalni radijus da ostane unutar slike
    max_r = min([center(1), center(2), height - center(1), width - center(2)]);

    % Kreiraj koordinatnu mrežu
    [XX, YY] = meshgrid(1:width, 1:height);

    % Rastojanje od centra
    dist_from_center = sqrt((XX - center(2)).^2 + (YY - center(1)).^2);

    % Kružna maska
    circle_mask = dist_from_center <= max_r;

    % Primeni masku
    masked_img = img;
    masked_img(~circle_mask) = NaN;

    % Broj piksela unutar kruga
    %num_inside_circle = nnz(circle_mask);  % broj elemenata koji su "true"
end
