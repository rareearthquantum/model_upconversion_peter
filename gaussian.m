function z = gaussian(a,b)

z = 1/(2*pi).*exp(-(a.^2+b.^2)/2);