function run_test(a, b, c, d, base_current, max_input, num_points)

    base_rate = raw_spike_rate(a, b, c, d, base_current);

    index = 1;
    input_range = linspace(-1,max_input,num_points);
    tuning_curve = zeros(num_points, 1);
    for input = input_range
        tuning_curve(index) = adjusted_spike_rate(a, b, c, d, base_current, base_rate, input);
        index = index + 1;
    end
    
    plot(input_range, tuning_curve, 'LineWidth',2)

end
