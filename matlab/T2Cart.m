function x = T2Cart(T)
x = [
    T(1:3,4);
    R2pqr(T(1:3,1:3));
];
end