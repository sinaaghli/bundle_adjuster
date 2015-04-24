function pqr = R2pqr( R )

cqsr = R(1,3);
cqsp = R(2,1);
cqcp = R(2,2);
cqcr = R(3,3);
sq  = -R(2,3);

p = atan2(cqsp, cqcp);
r = atan2(cqsr, cqcr);
if cqcr ~= 0
    q = atan(sq / cqcr * cos(r));
else
    q = atan(sq / cqsr * sin(r));
end

pqr = [p q r]';

end
