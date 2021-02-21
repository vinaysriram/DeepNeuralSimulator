function r = raw_spike_rate(a, b, c, d, input)

    u = d; v = c; dt = 0.01; T = 0;
    
    while v < 30
        dv = (0.04 * v^2 + 5.0*v + 140 - u + input) * dt;
        du = (a*(b*v-u)) * dt;
        v = v + dv;
        u = u + du;
        T = T + dt;
    end
    
    r = 1/T;
    
end