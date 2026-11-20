#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "rf_transport.hpp"

namespace aegis::rt {

/// Bit-True Fixed-Point Q1.15 Real-Time DRFM Core.
/// Matches FPGA RTL (fpga/rtl/drfm_core_q15.sv) and Python golden model (aegis_rt_bridge):
///   1. Configurable N-bit ADC/DAC quantization (1..15 bits)
///   2. Power-of-two circular BRAM delay line (RGPO)
///   3. 32-bit phase-accumulator NCO with 1024-entry Q1.15 sin/cos LUT (VGPO)
class DRFMCoreQ15 {
public:
    static constexpr std::size_t kLutBits = 10;
    static constexpr std::size_t kLutSize = 1u << kLutBits;
    static constexpr double kPi = 3.14159265358979323846;

    explicit DRFMCoreQ15(std::size_t max_delay_pow2 = 65536, int adc_bits = 8)
        : delay_mask_(max_delay_pow2 - 1),
          delay_mem_(max_delay_pow2, IQ16{0, 0}),
          adc_bits_(std::clamp(adc_bits, 1, 15)) {
        init_nco_lut();
    }

    void set_delay_samples(std::uint32_t delay_samples) noexcept {
        delay_samples_ = delay_samples & static_cast<std::uint32_t>(delay_mask_);
    }

    void set_doppler_hz(double fd_hz, double fs_hz) noexcept {
        // 32-bit tuning word: round(fd / fs * 2^32)
        const double ratio = fd_hz / fs_hz;
        phase_inc_ = static_cast<std::uint32_t>(static_cast<std::int64_t>(std::llround(ratio * 4294967296.0)));
    }

    void set_adc_bits(int bits) noexcept {
        adc_bits_ = std::clamp(bits, 1, 15);
    }

    /// Process a contiguous block of Q1.15 IQ samples with zero heap allocations.
    void process_block(const IQ16* in, IQ16* out, std::size_t n) noexcept {
        const std::uint32_t shift = static_cast<std::uint32_t>(16 - adc_bits_);
        const int16_t mask = (adc_bits_ == 1) ? 0 : static_cast<int16_t>(~((1u << shift) - 1u));

        for (std::size_t k = 0; k < n; ++k) {
            // 1. N-bit DRFM ADC Quantization
            IQ16 q;
            if (adc_bits_ == 1) {
                q.i = (in[k].i >= 0) ? 23170 : -23170; // +/- 0.7071 in Q1.15
                q.q = (in[k].q >= 0) ? 23170 : -23170;
            } else {
                q.i = static_cast<int16_t>(in[k].i & mask);
                q.q = static_cast<int16_t>(in[k].q & mask);
            }

            // 2. Circular BRAM Delay Line (RGPO)
            delay_mem_[wr_ptr_] = q;
            const std::size_t rd_ptr = (wr_ptr_ - delay_samples_) & delay_mask_;
            const IQ16 d = delay_mem_[rd_ptr];
            wr_ptr_ = (wr_ptr_ + 1) & delay_mask_;

            // 3. 32-bit Phase-Accumulator NCO Complex Mixer (VGPO)
            const std::size_t lut_idx = (phase_acc_ >> (32 - kLutBits)) & (kLutSize - 1);
            const int32_t c = cos_lut_[lut_idx];
            const int32_t s = sin_lut_[lut_idx];
            phase_acc_ += phase_inc_;

            const int32_t oi = (static_cast<int32_t>(d.i) * c - static_cast<int32_t>(d.q) * s) >> 15;
            const int32_t oq = (static_cast<int32_t>(d.i) * s + static_cast<int32_t>(d.q) * c) >> 15;

            out[k].i = static_cast<int16_t>(std::clamp<int32_t>(oi, -32768, 32767));
            out[k].q = static_cast<int16_t>(std::clamp<int32_t>(oq, -32768, 32767));
        }
    }

private:
    void init_nco_lut() {
        for (std::size_t k = 0; k < kLutSize; ++k) {
            const double ang = 2.0 * kPi * static_cast<double>(k) / static_cast<double>(kLutSize);
            cos_lut_[k] = static_cast<int16_t>(std::llround(std::cos(ang) * 32767.0));
            sin_lut_[k] = static_cast<int16_t>(std::llround(std::sin(ang) * 32767.0));
        }
    }

    std::size_t delay_mask_;
    std::vector<IQ16> delay_mem_;
    std::size_t wr_ptr_ = 0;
    std::uint32_t delay_samples_ = 0;
    std::uint32_t phase_acc_ = 0;
    std::uint32_t phase_inc_ = 0;
    int adc_bits_;

    std::array<int16_t, kLutSize> cos_lut_{};
    std::array<int16_t, kLutSize> sin_lut_{};
};

} // namespace aegis::rt
