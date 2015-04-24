function T = Cart2T(x)
T = [
    pqr2R(x(4:6,1)), x(1:3,1);
    0, 0, 0, 1
];
end