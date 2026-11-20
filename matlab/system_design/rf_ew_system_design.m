function results = rf_ew_system_design(outDir)
% RF_EW_SYSTEM_DESIGN  AEGIS-DRFM Phase 1 MATLAB System Design Script
%   Computes Radar Range Equation, SNR vs Range, Self-Screening Jammer (SSJ)
%   J/S Ratio, Burn-Through Range, and exports a unified system_config.json
%   consumed by the Python core-dsp and drfm-ecm engines.
%
%   Usage:
%       results = rf_ew_system_design();
%       results = rf_ew_system_design('matlab/outputs');

    if nargin < 1 || isempty(outDir)
        thisFile = mfilename('fullpath');
        repoRoot = fileparts(fileparts(fileparts(thisFile)));
        outDir = fullfile(repoRoot, 'matlab', 'outputs');
    end
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    %% 1. Physical Constants & X-Band Fire-Control / Tracking Radar Parameters
    c   = 299792458.0;       % Speed of light (m/s)
    kB  = 1.380649e-23;      % Boltzmann constant (J/K)
    T0  = 290.0;             % Standard noise temperature (K)

    radar.name            = 'AEGIS-XBand-Tracker';
    radar.fc_hz           = 10.0e9;      % Carrier frequency: 10 GHz (X-band)
    radar.lambda_m        = c / radar.fc_hz;
    radar.pt_w            = 25e3;        % Peak transmit power: 25 kW
    radar.gt_db           = 32.0;        % TX antenna gain (dBi)
    radar.gr_db           = 32.0;        % RX antenna gain (dBi)
    radar.nf_db           = 4.0;         % Receiver noise figure (dB)
    radar.loss_db         = 5.0;         % Total radar system losses (dB)
    radar.bandwidth_hz    = 20.0e6;      % LFM Chirp bandwidth: 20 MHz
    radar.pulse_width_s   = 10.0e-6;     % Pulse width: 10 us (BT = 200 -> 23 dB gain)
    radar.prf_hz          = 5000.0;      % PRF: 5 kHz (Unambiguous range = 30 km)
    radar.num_pulses      = 64;          % Coherent Processing Interval (CPI) pulses
    radar.fs_hz           = 50.0e6;      % Baseband complex sampling rate: 50 MSPS
    radar.target_rcs_m2   = 5.0;         % Target RCS: 5 m^2 (7 dBsm)
    radar.target_range_m  = 12000.0;     % Nominal target range: 12 km
    radar.target_vel_mps  = -220.0;      % Radial velocity: -220 m/s (closing)

    %% 2. DRFM Self-Screening Jammer (SSJ) Parameters
    jammer.erp_w          = 500.0;       % Jammer ERP (Pj * Gj): 500 W (27 dBW)
    jammer.bandwidth_hz   = 20.0e6;      % Matched to radar bandwidth (coherent DRFM)
    jammer.loss_db        = 3.0;         % Polarization + DRFM internal loss (dB)
    jammer.adc_bits       = 8;           % Nominal DRFM ADC/DAC bit depth
    jammer.loop_delay_ns  = 120.0;       % Hardware capture-to-replay latency (ns)

    %% 3. Range Sweep: Radar SNR & Jammer-to-Signal Ratio (J/S)
    R_km = linspace(1.0, 30.0, 300);
    R_m  = R_km * 1e3;

    Gt = 10^(radar.gt_db / 10);
    Gr = 10^(radar.gr_db / 10);
    Fn = 10^(radar.nf_db / 10);
    Lr = 10^(radar.loss_db / 10);
    Lj = 10^(jammer.loss_db / 10);

    % Single-pulse received target power (Radar Range Equation)
    Pr_target = (radar.pt_w * Gt * Gr * (radar.lambda_m^2) * radar.target_rcs_m2) ...
                ./ ((4*pi)^3 .* (R_m.^4) .* Lr);

    % Thermal noise power across receiver bandwidth
    Pn = kB * T0 * radar.bandwidth_hz * Fn;

    % Pulse compression gain (BT) and coherent integration gain (N_pulses)
    bt_gain  = radar.bandwidth_hz * radar.pulse_width_s;
    cpi_gain = radar.num_pulses;

    snr_single_db = 10 * log10(Pr_target / Pn);
    snr_pc_db     = snr_single_db + 10 * log10(bt_gain);
    snr_cpi_db    = snr_pc_db + 10 * log10(cpi_gain);

    % Self-Screening Jammer (SSJ) received power at radar (1/R^2 law)
    % Bandwidth ratio accounts for coherent DRFM matching radar waveform
    bw_ratio = min(1.0, radar.bandwidth_hz / jammer.bandwidth_hz);
    Pr_jammer = (jammer.erp_w * Gr * (radar.lambda_m^2) * bw_ratio) ...
                ./ ((4*pi)^2 .* (R_m.^2) .* Lj);

    % J/S Ratio (dB)
    jsr_db = 10 * log10(Pr_jammer ./ Pr_target);

    % Burn-through range where J/S = 0 dB (target echo equals jammer power)
    % Pr_target / Pr_jammer = 1 => R_bt^2 = (Pt * Gt * sigma * Lj) / (4*pi * ERP_j * Lr * bw_ratio)
    R_bt_m = sqrt((radar.pt_w * Gt * radar.target_rcs_m2 * Lj) ...
                  / (4 * pi * jammer.erp_w * Lr * bw_ratio));

    % Evaluate nominal target range values
    [~, idx_nom] = min(abs(R_m - radar.target_range_m));
    nom_snr_single_db = snr_single_db(idx_nom);
    nom_snr_pc_db     = snr_pc_db(idx_nom);
    nom_snr_cpi_db    = snr_cpi_db(idx_nom);
    nom_jsr_db        = jsr_db(idx_nom);

    %% 4. Export Unified Configuration & Golden Reference Metrics to JSON
    cfg.radar  = radar;
    cfg.jammer = jammer;
    cfg.derived.bt_gain_db         = 10 * log10(bt_gain);
    cfg.derived.cpi_gain_db        = 10 * log10(cpi_gain);
    cfg.derived.unambiguous_rng_m  = c / (2 * radar.prf_hz);
    cfg.derived.range_res_m        = c / (2 * radar.bandwidth_hz);
    cfg.derived.doppler_res_hz     = radar.prf_hz / radar.num_pulses;
    cfg.derived.vel_res_mps        = cfg.derived.doppler_res_hz * radar.lambda_m / 2;
    cfg.derived.burn_through_m     = R_bt_m;
    cfg.derived.nom_snr_single_db  = nom_snr_single_db;
    cfg.derived.nom_snr_pc_db      = nom_snr_pc_db;
    cfg.derived.nom_snr_cpi_db     = nom_snr_cpi_db;
    cfg.derived.nom_jsr_db         = nom_jsr_db;

    jsonText = jsonencode(cfg, 'PrettyPrint', true);
    fid = fopen(fullfile(outDir, 'system_config.json'), 'w');
    fwrite(fid, jsonText, 'char');
    fclose(fid);

    %% 5. Generate System Design Plot
    fig = figure('Visible', 'off', 'Position', [100, 100, 1100, 480], 'Color', 'w');

    subplot(1, 2, 1);
    plot(R_km, snr_single_db, '--', 'LineWidth', 1.4, 'DisplayName', 'Single-Pulse SNR'); hold on;
    plot(R_km, snr_pc_db, '-', 'LineWidth', 1.6, 'DisplayName', 'Pulse-Compressed SNR (+BT)');
    plot(R_km, snr_cpi_db, '-', 'LineWidth', 2.0, 'DisplayName', 'Coherent CPI SNR (+N_{pulses})');
    yline(13.0, ':r', 'LineWidth', 1.2, 'DisplayName', 'Detection Threshold (~13 dB)');
    xline(radar.target_range_m / 1e3, '-.k', 'Nominal Target (12 km)');
    grid on; xlabel('Range (km)'); ylabel('SNR (dB)');
    title('X-Band Radar Link Budget & Processing Gain');
    legend('Location', 'southwest');

    subplot(1, 2, 2);
    plot(R_km, jsr_db, 'LineWidth', 2.0, 'Color', [0.85 0.2 0.1], 'DisplayName', 'DRFM SSJ J/S Ratio'); hold on;
    yline(0, '--k', 'LineWidth', 1.2, 'DisplayName', 'Burn-Through (J/S = 0 dB)');
    xline(R_bt_m / 1e3, ':b', sprintf('R_{BT} = %.2f km', R_bt_m / 1e3), 'LineWidth', 1.4);
    xline(radar.target_range_m / 1e3, '-.k', sprintf('Target: %.1f dB J/S', nom_jsr_db));
    grid on; xlabel('Range (km)'); ylabel('J/S Ratio (dB)');
    title('Self-Screening DRFM Jammer-to-Signal Ratio');
    legend('Location', 'southeast');

    exportgraphics(fig, fullfile(outDir, 'rf_ew_system_design.png'), 'Resolution', 150);
    close(fig);

    fprintf('[MATLAB] System design complete.\n');
    fprintf('  Range Res: %.2f m | Vel Res: %.2f m/s | Unambiguous R: %.1f km\n', ...
            cfg.derived.range_res_m, cfg.derived.vel_res_mps, cfg.derived.unambiguous_rng_m/1e3);
    fprintf('  At 12 km -> SNR(1-pulse)=%.2f dB, SNR(PC)=%.2f dB, SNR(CPI)=%.2f dB, J/S=%.2f dB\n', ...
            nom_snr_single_db, nom_snr_pc_db, nom_snr_cpi_db, nom_jsr_db);
    fprintf('  Burn-Through Range (J/S=0 dB): %.2f km\n', R_bt_m / 1e3);

    results = cfg;
end
