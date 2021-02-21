clear all
close all
hold on
grid on

base_current = 10.0;
max_input = 10.0;
num_points = 1000;

a_rs = 0.02; b_rs = 0.20; c_rs = -65; d_rs = 8.00; 
a_ib = 0.02; b_ib = 0.20; c_ib = -55; d_ib = 4.00;
a_ch = 0.02; b_ch = 0.20; c_ch = -50; d_ch = 2.00; 
a_fs = 0.10; b_fs = 0.20; c_fs = -65; d_fs = 2.00;
a_rz = 0.10; b_rz = 0.26; c_rz = -65; d_rz = 2.00;
a_tc = 0.02; b_tc = 0.25; c_tc = -65; d_tc = 0.05;
a_ls = 0.02; b_ls = 0.25; c_ls = -65; d_ls = 2.00;

generate_tuning_curve(a_rs, b_rs, c_rs, d_rs, base_current, max_input, num_points);
generate_tuning_curve(a_ib, b_ib, c_ib, d_ib, base_current, max_input, num_points);
generate_tuning_curve(a_ch, b_ch, c_ch, d_ch, base_current, max_input, num_points);
generate_tuning_curve(a_fs, b_fs, c_fs, d_fs, base_current, max_input, num_points);
generate_tuning_curve(a_rz, b_rz, c_rz, d_rz, base_current, max_input, num_points);
generate_tuning_curve(a_tc, b_tc, c_tc, d_tc, base_current, max_input, num_points);
generate_tuning_curve(a_ls, b_ls, c_ls, d_ls, base_current, max_input, num_points);

legend({'RS', 'IB', 'CH', 'FS', 'RZ', 'TC', 'LS'}, 'Location','northwest');
xlabel('Input')
ylabel('Neuron Firing Rate')
title('Comparison of Neural Firing Rates')
xlim([-1 max_input])
