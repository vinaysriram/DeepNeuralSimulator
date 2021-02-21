function r = adjusted_spike_rate(a, b, c, d, base_current, base_rate, input)
    r = 0;
    if input > 0
        r = raw_spike_rate(a, b, c, d, input + base_current) - base_rate;
    end
end
       

  