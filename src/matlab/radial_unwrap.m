function transformed_img = radial_unwrap(img, num_angles, num_radii, center)
% Transformiše sliku iz polarnih pravaca u pravougaoni oblik
%
% Ulazi:
% - img: ulazna slika (2D matrica)
% - num_angles: broj pravaca (ugao rezolucija)
% - num_radii: broj rastojanja (koliko uzoraka po pravcu)
% - center: [yc, xc] – centar slike
%
% Izlaz:
% - transformed_img: slika gde su kolone pravci, redovi rastojanja

% Uglovi: 0 do 2*pi
theta = linspace(0, 2*pi, num_angles + 1);
theta(end) = [];  % poslednji se poklapa s prvim

% Rastojanja: ravnomerno od 0 do max
max_r = min([center(1), center(2), size(img,1)-center(1), size(img,2)-center(2)]);
r = linspace(0, max_r, num_radii);

% Inicijalizuj izlaznu sliku
transformed_img = zeros(num_radii, num_angles);

% Pretvori ulaznu sliku u double za preciznu interpolaciju
img = double(img);

% Petlja po uglovima i rastojanjima
for i = 1:num_angles
    for j = 1:num_radii
        x = center(2) + r(j) * cos(theta(i));
        y = center(1) + r(j) * sin(theta(i));
        
        % Interpolacija: ako je tačka van slike, koristi 0
        transformed_img(j, i) = interp2(img, x, y, 'linear', 0);
    end
end
end

