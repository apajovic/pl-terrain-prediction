function img = radial_wrap(transformed_img, img_size, center)
% RADIAL_WRAP - Rekonstruiše sliku iz unwrap-ovanog oblika uz interpolaciju unutar kruga
%
% Ulazi:
%   transformed_img - matrica: kolone = pravci, redovi = rastojanje od centra
%   img_size - veličina originalne slike [visina, širina]
%   center - koordinata centra [yc, xc]
%
% Izlaz:
%   img - rekonstruisana slika

[num_radii, num_angles] = size(transformed_img);

% Definišemo uglove (theta) i rastojanja (r)
theta = linspace(0, 2*pi, num_angles + 1);  
theta(end) = [];  % Izbacujemo duplikat na kraju
max_r = min([center(1), center(2), img_size(1)-center(1), img_size(2)-center(2)]);
r = linspace(0, max_r, num_radii);

% Inicijalizacija prazne slike i mape težina
img = zeros(img_size);
weight = zeros(img_size);

% Popunjavamo poznate vrednosti u slici
for i = 1:num_angles
    for j = 1:num_radii
        x = center(2) + r(j) * cos(theta(i));
        y = center(1) + r(j) * sin(theta(i));

        xi = round(x);
        yi = round(y);

        if xi >= 1 && xi <= img_size(2) && yi >= 1 && yi <= img_size(1)
            img(yi, xi) = img(yi, xi) + transformed_img(j, i);
            weight(yi, xi) = weight(yi, xi) + 1;
        end
    end
end

% Izračunavanje prosečne vrednosti
weight(weight == 0) = NaN;  % da ne bi delili nulom
img = img ./ weight;

% Pravimo kružnu masku
[XX, YY] = meshgrid(1:img_size(2), 1:img_size(1));
dist_from_center = sqrt((XX - center(2)).^2 + (YY - center(1)).^2);
circle_mask = dist_from_center <= max_r;

% Interpolacija pomoću scatteredInterpolant unutar kruga
[Y_known, X_known] = find(~isnan(img));        % Poznate pozicije
V_known = img(~isnan(img));                    % Njihove vrednosti
F = scatteredInterpolant(X_known, Y_known, V_known, 'natural', 'none');  % Interpolacija

interp_values = F(XX, YY);                     % Interpolacija svuda
img(circle_mask & isnan(img)) = interp_values(circle_mask & isnan(img));  % Popuna unutar kruga

% Van kruga postavljamo nulu
img(~circle_mask) = 0;
end
