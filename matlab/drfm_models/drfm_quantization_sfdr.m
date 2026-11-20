function report = drfm_quantization_sfdr(outDir)
% DRFM_QUANTIZATION_SFDR  Models DRFM ADC/DAC bit-depth impact on SQNR, ENOB,
%   and Spurious-Free Dynamic Range (SFDR) for coherent RF replay.
%   Generates reference metrics for Python & future FPGA fixed-point validation.

    if nargin < 1 || isempty(outDir)
        thisFile = mfilename('fullpath');
        repoRoot = fileparts(fileparts(fileparts(thisFile)));
        outDir = fullfile(repoRoot, 'matlab', 'outputs');
    end
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    %% 1. Coherent Test Tone Setup (Non-integer bin ratio avoided via coherent sampling)
    fs    = 50e6;
    N     = 4096;
    k_bin = 317;              % Prime FFT bin for clean harmonic folding analysis
    f0    = k_bin * fs / N;
    n     = (0:N-1);
    t     = n / fs;

    % Full-scale complex baseband tone (0.98 FS to avoid clipping)
    A     = 0.98;
    x     = A * exp(1j * 2 * pi * f0 * t);

    bitDepths = [1, 2, 4, 8, 12];
    nB = numel(bitDepths);

    sqnr_meas_db  = zeros(1, nB);
    sqnr_theo_db  = zeros(1, nB);
    sfdr_meas_db  = zeros(1, nB);
    enob_meas     = zeros(1, nB);
    spectra_db    = zeros(nB, N);

    w = hann(N).';
    w = w / sum(w);

    for i = 1:nB
        B = bitDepths(i);
        xq = quantize_complex_drfm(x, B);

        % Normalize replay amplitude for fair SQNR comparison
        scale = (xq * x') / (xq * xq');
        xq_norm = xq * scale;

        err = xq_norm - x;
        p_sig = mean(abs(x).^2);
        p_err = mean(abs(err).^2);

        sqnr_meas_db(i) = 10 * log10(p_sig / max(p_err, 1e-18));
        sqnr_theo_db(i) = 6.02 * B + 1.76; % Ideal full-scale sinusoid reference
        enob_meas(i)    = (sqnr_meas_db(i) - 1.76) / 6.02;

        % Spectral SFDR measurement
        X = fftshift(fft(xq .* w));
        mag_db = 20 * log10(abs(X) + 1e-15);
        mag_db = mag_db - max(mag_db);
        spectra_db(i, :) = mag_db;

        % Mask fundamental mainlobe (+/- 4 bins around peak) to find strongest spur
        [~, pk_idx] = max(mag_db);
        spur_mask = true(1, N);
        lo = max(1, pk_idx - 4);
        hi = min(N, pk_idx + 4);
        spur_mask(lo:hi) = false;
        sfdr_meas_db(i) = -max(mag_db(spur_mask));
    end

    %% 2. Export Golden Reference JSON
    report.fs_hz        = fs;
    report.n_fft        = N;
    report.tone_hz      = f0;
    report.bit_depths   = bitDepths;
    report.sqnr_meas_db = sqnr_meas_db;
    report.sqnr_theo_db = sqnr_theo_db;
    report.sfdr_meas_db = sfdr_meas_db;
    report.enob_meas    = enob_meas;

    jsonText = jsonencode(report, 'PrettyPrint', true);
    fid = fopen(fullfile(outDir, 'drfm_quantization_ref.json'), 'w');
    fwrite(fid, jsonText, 'char');
    fclose(fid);

    %% 3. Plot Spectra & Metrics
    f_mhz = ((-N/2):(N/2-1)) * (fs / N) / 1e6;
    fig = figure('Visible', 'off', 'Position', [100, 100, 1200, 500], 'Color', 'w');

    subplot(1, 2, 1);
    hold on;
    colors = lines(nB);
    for i = [1, 3, 4, 5] % Show 1, 4, 8, 12 bit spectra
        plot(f_mhz, spectra_db(i, :), 'LineWidth', 1.1, 'Color', colors(i, :), ...
             'DisplayName', sprintf('%d-bit (SFDR=%.1f dB)', bitDepths(i), sfdr_meas_db(i)));
    end
    grid on; ylim([-110, 5]);
    xlabel('Baseband Frequency (MHz)'); ylabel('Normalized Magnitude (dBc)');
    title('DRFM Replay Spectrum vs. ADC/DAC Bit Depth');
    legend('Location', 'southwest');

    subplot(1, 2, 2);
    plot(bitDepths, sqnr_meas_db, '-o', 'LineWidth', 1.8, 'DisplayName', 'Measured SQNR (dB)'); hold on;
    plot(bitDepths, sqnr_theo_db, '--k', 'LineWidth', 1.3, 'DisplayName', '6.02B + 1.76 Theoretical');
    plot(bitDepths, sfdr_meas_db, '-s', 'LineWidth', 1.8, 'DisplayName', 'Measured SFDR (dBc)');
    grid on; xlabel('DRFM Quantization Bit Depth (B)'); ylabel('dB / dBc');
    title('DRFM Quantization SQNR & SFDR Trade-off');
    legend('Location', 'northwest');

    exportgraphics(fig, fullfile(outDir, 'drfm_quantization_sfdr.png'), 'Resolution', 150);
    close(fig);

    fprintf('[MATLAB] DRFM Quantization SFDR Analysis complete:\n');
    for i = 1:nB
        fprintf('  %2d-bit -> SQNR: %6.2f dB | SFDR: %6.2f dBc | ENOB: %5.2f\n', ...
                bitDepths(i), sqnr_meas_db(i), sfdr_meas_db(i), enob_meas(i));
    end
end

function xq = quantize_complex_drfm(x, bits)
% Uniform mid-riser/mid-tread I/Q quantizer (1-bit = sign/hard-limiter DRFM)
    if bits == 1
        qi = sign(real(x)); qi(qi == 0) = 1;
        qq = sign(imag(x)); qq(qq == 0) = 1;
        xq = (qi + 1j * qq) / sqrt(2);
    else
        levels = 2^bits;
        step   = 2.0 / levels;
        ri = max(min(real(x), 1.0 - step/2), -1.0);
        rq = max(min(imag(x), 1.0 - step/2), -1.0);
        qi = round(ri / step) * step;
        qq = round(rq / step) * step;
        xq = qi + 1j * qq;
    end
end
