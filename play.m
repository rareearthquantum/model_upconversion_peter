c = 10;
mygrid = @(x,y) ndgrid((-x:x/c:x),(-y:y/c:y))*2;
[x] = mygrid(pi,2*pi);

z = sin(x);
% figure(11);mesh(x,y,z)

% [a1,a2] = arrayfun(@(x,y) mygrid(x,y), [1],[4])