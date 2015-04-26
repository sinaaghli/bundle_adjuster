function R = pqr2R( pqr )
% r = pqr(1);
% p = pqr(2);
% y = pqr(3);
% Rz = [cos(r) -sin(r) 0; sin(r) cos(r) 0; 0 0 1];
% Rx = [1 0 0; 0 cos(p) -sin(p); 0 sin(p) cos(p)];
% Ry = [cos(y) 0 sin(y); 0 1 0; -sin(y) 0 cos(y)];
% R = Ry * Rx * Rz;
% r = pqr(1);
% p = pqr(2);
% y = pqr(3);

% syms p q r s(x) c(x)
% subs(subs(pqr2R([p;q;r]),{sin(p),sin(q),sin(r)},{s(1),s(2),s(3)}), {cos(p),cos(q),cos(r)},{c(1),c(2),c(3)})
c = cos(pqr);
s = sin(pqr);
R = ...
[ c(1)*c(3) + s(1)*s(2)*s(3), c(1)*s(2)*s(3) - c(3)*s(1), c(2)*s(3); ...
                   c(2)*s(1),                  c(1)*c(2),     -s(2); ...
  c(3)*s(1)*s(2) - c(1)*s(3), s(1)*s(3) + c(1)*c(3)*s(2), c(2)*c(3)];

end
