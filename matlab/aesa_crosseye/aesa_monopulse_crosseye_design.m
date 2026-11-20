function report = aesa_monopulse_crosseye_design(outDir)
% AESA_MONOPULSE_CROSSEYE_DESIGN  MATLAB R2025b System Design for 16-Element
%   X-Band ULA AESA Monopulse Radar vs. Retrodirective Cross-Eye DRFM Jammer.
%   Exports golden reference JSON for Python aegis-aesa cross-validation.

    if nargin < 1 || isempty(outDir)
        thisFile = mfilename('fullpath');
        repoRoot = fileparts(fileparts(fileparts(thisFile)));
        outDir = fullfile(repoRoot, 'matlab', 'outputs');
    end
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    c      = 299792458.0;
    fc     = 10.0e9;
    lambda = c / fc;
    N      = 16;
    d      = 0.5 * lambda;

    % Symmetric element coordinates
    pos = ((0:N-1).' - 0.5 * (N - 1)) * d;

    % Sum and Difference weights steered to 0 deg
    diff_sign = [-ones(N/2, 1); ones(N/2, 1)];
    w_sum  = ones(N, 1) / N;
    w_diff = diff_sign / N;

    steer = @(th_deg) exp(1j * 2 * pi * (pos / lambda) * sind(th_deg));

    % Calibrate monopulse slope k_m near boresight
    eps_d = 0.05;
    a_p = steer(eps_d); a_m = steer(-eps_d);
    m_p = (w_diff' * a_p) / (1j * (w_sum' * a_p));
    m_m = (w_diff' * a_m) / (1j * (w_sum' * a_m));
    k_m = real(m_p - m_m) / (2 * eps_d);

    % Cross-Eye Operating Point: baseline=15 m, R=3000 m, a=0.90, phi=178 deg, JSR=25 dB
    baseline_m = 15.0;
    R_m        = 3000.0;
    a_ratio    = 0.90;
    phi_deg    = 178.0;
    jsr_db     = 25.0;

    theta_e_deg = rad2deg(atan2(0.5 * baseline_m, R_m));
    a1 = steer(-theta_e_deg);
    a2 = steer(+theta_e_deg);
    a0 = steer(0.0);

    jam_amp = 10^(jsr_db / 20);
    w_ce    = a_ratio * exp(1j * deg2rad(phi_deg));
    snap    = a0 + jam_amp * (a1 + w_ce * a2);

    S_val = w_sum'  * snap;
    D_val = w_diff' * snap;
    M_val = D_val / (1j * S_val);

    est_angle_deg = real(M_val) / k_m;
    quad_ratio    = abs(imag(M_val));

    % Analytical Cross-Eye Gain
    gc = (1 - w_ce) / (1 + w_ce);

    % Export Golden Reference JSON
    report.num_elements   = N;
    report.fc_hz          = fc;
    report.k_monopulse    = k_m;
    report.theta_e_deg    = theta_e_deg;
    report.crosseye_gc_re = real(gc);
    report.crosseye_gc_im = imag(gc);
    report.est_angle_deg  = est_angle_deg;
    report.quad_ratio     = quad_ratio;

    fid = fopen(fullfile(outDir, 'aesa_crosseye_ref.json'), 'w');
    fwrite(fid, jsonencode(report, 'PrettyPrint', true), 'char');
    fclose(fid);

    % Plot MATLAB Verification Figure
    angles = linspace(-20, 20, 401);
    S_pat = zeros(size(angles));
    D_pat = zeros(size(angles));
    for k = 1:numel(angles)
        ak = steer(angles(k));
        S_pat(k) = abs(w_sum' * ak);
        D_pat(k) = abs(w_diff' * ak);
    end

    phi_sw = linspace(170, 190, 201);
    gc_re_sw = zeros(size(phi_sw));
    gc_im_sw = zeros(size(phi_sw));
    for k = 1:numel(phi_sw)
        wk = a_ratio * exp(1j * deg2rad(phi_sw(k)));
        gck = (1 - wk) / (1 + wk);
        gc_re_sw(k) = real(gck);
        gc_im_sw(k) = abs(imag(gck));
    end

    fig = figure('Visible', 'off', 'Position', [100, 100, 1100, 460], 'Color', 'w');
    subplot(1, 2, 1);
    plot(angles, 20*log10(S_pat + 1e-12), 'b-', 'LineWidth', 1.8, 'DisplayName', 'Sum (\Sigma)'); hold on;
    plot(angles, 20*log10(D_pat + 1e-12), 'r--', 'LineWidth', 1.6, 'DisplayName', 'Diff (\Delta)');
    ylim([-45 2]); grid on; xlabel('Angle (deg)'); ylabel('Gain (dB)');
    title('MATLAB: 16-Element AESA Monopulse Beams'); legend('Location', 'northeast');

    subplot(1, 2, 2);
    yyaxis left;
    plot(phi_sw, gc_re_sw, 'LineWidth', 2.0, 'DisplayName', 'Re(G_C) Pull-Off Gain');
    ylabel('Cross-Eye Pull-Off Gain Re(G_C)');
    yyaxis right;
    plot(phi_sw, gc_im_sw, '--', 'LineWidth', 1.8, 'DisplayName', '|Im(G_C)| Quadrature Signature');
    ylabel('Quadrature Signature |Im(G_C)|');
    grid on; xlabel('Cross-Eye Relative Phase (deg)');
    title('MATLAB: Cross-Eye Gain & Quadrature ECCM Signature (a=0.90)');

    exportgraphics(fig, fullfile(outDir, 'aesa_monopulse_crosseye.png'), 'Resolution', 150);
    close(fig);

    fprintf('[MATLAB] AESA/Monopulse Cross-Eye design complete.\n');
    fprintf('  k_m = %.4f / deg | theta_e = %.4f deg | Apparent Angle = %+.3f deg | Quad = %.4f\n', ...
            k_m, theta_e_deg, est_angle_deg, quad_ratio);
end
