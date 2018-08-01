function [ w ] = simpsonw2(s)
%SIMPSONW gives the weights needed for simpsons rule for numerical
%integration


if rem(s(1),2)~=1 || rem(s(2),2)~=1
   error('n and m  must be odd')
end

u = ones(1,s(1));
u(1:2:end) = 2;
u(2:2:end) = 4;
u(1) = 1;
u(end)=1;


v = ones(1,s(2));
v(1:2:end) = 2;
v(2:2:end) = 4;
v(1) = 1;
v(end)=1;

w = u'*v/9;


end

