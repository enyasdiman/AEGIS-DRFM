#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace aegis::rt {

/// Fixed-point Q1.15 complex IQ sample (matches PlutoSDR/USRP sc16 & FPGA AXI-Stream).
struct alignas(4) IQ16 {
    int16_t i;
    int16_t q;
};

struct RFStreamConfig {
    double sample_rate_hz = 50.0e6;
    double center_freq_hz = 10.0e9;
    double rx_gain_db     = 40.0;
    double tx_gain_db     = 30.0;
    std::size_t mtu_samples = 4096;
};

/// Hardware-agnostic RF Transport abstraction.
/// Swap SimulatedLoopbackTransport -> SoapySDRTransport without touching DRFM DSP code.
class IRFTransport {
public:
    virtual ~IRFTransport() = default;
    virtual bool open(const RFStreamConfig& cfg) = 0;
    virtual std::size_t read_iq(IQ16* dst, std::size_t max_samples) = 0;
    virtual std::size_t write_iq(const IQ16* src, std::size_t num_samples) = 0;
    virtual void close() = 0;
};

/// Zero-hardware loopback transport for offline C++ real-time validation & latency testing.
class SimulatedLoopbackTransport final : public IRFTransport {
public:
    bool open(const RFStreamConfig& cfg) override {
        cfg_ = cfg;
        rx_buf_.assign(cfg.mtu_samples, IQ16{0, 0});
        tx_buf_.assign(cfg.mtu_samples, IQ16{0, 0});
        return true;
    }

    void inject_rx_block(const std::vector<IQ16>& block) { rx_buf_ = block; }
    const std::vector<IQ16>& last_tx_block() const noexcept { return tx_buf_; }

    std::size_t read_iq(IQ16* dst, std::size_t max_samples) override {
        const std::size_t n = (max_samples < rx_buf_.size()) ? max_samples : rx_buf_.size();
        for (std::size_t k = 0; k < n; ++k) dst[k] = rx_buf_[k];
        return n;
    }

    std::size_t write_iq(const IQ16* src, std::size_t num_samples) override {
        tx_buf_.assign(src, src + num_samples);
        return num_samples;
    }

    void close() override {}

private:
    RFStreamConfig cfg_{};
    std::vector<IQ16> rx_buf_;
    std::vector<IQ16> tx_buf_;
};

#ifdef AEGIS_HAS_SOAPYSDR
#include <SoapySDR/Device.hpp>
#include <SoapySDR/Formats.hpp>

/// Ready-to-use SoapySDR hardware transport (PlutoSDR, BladeRF, USRP, HackRF).
class SoapySDRTransport final : public IRFTransport {
public:
    explicit SoapySDRTransport(std::string driver_args = "driver=plutosdr")
        : args_(std::move(driver_args)) {}

    bool open(const RFStreamConfig& cfg) override {
        dev_ = SoapySDR::Device::make(args_);
        if (!dev_) return false;
        dev_->setSampleRate(SOAPY_SDR_RX, 0, cfg.sample_rate_hz);
        dev_->setSampleRate(SOAPY_SDR_TX, 0, cfg.sample_rate_hz);
        dev_->setFrequency(SOAPY_SDR_RX, 0, cfg.center_freq_hz);
        dev_->setFrequency(SOAPY_SDR_TX, 0, cfg.center_freq_hz);
        dev_->setGain(SOAPY_SDR_RX, 0, cfg.rx_gain_db);
        dev_->setGain(SOAPY_SDR_TX, 0, cfg.tx_gain_db);
        rx_stream_ = dev_->setupStream(SOAPY_SDR_RX, SOAPY_SDR_CS16);
        tx_stream_ = dev_->setupStream(SOAPY_SDR_TX, SOAPY_SDR_CS16);
        dev_->activateStream(rx_stream_);
        dev_->activateStream(tx_stream_);
        return true;
    }

    std::size_t read_iq(IQ16* dst, std::size_t max_samples) override {
        void* buffs[] = {dst};
        int flags = 0;
        long long time_ns = 0;
        int ret = dev_->readStream(rx_stream_, buffs, max_samples, flags, time_ns, 100000);
        return (ret > 0) ? static_cast<std::size_t>(ret) : 0;
    }

    std::size_t write_iq(const IQ16* src, std::size_t num_samples) override {
        const void* buffs[] = {src};
        int flags = 0;
        int ret = dev_->writeStream(tx_stream_, buffs, num_samples, flags, 0, 100000);
        return (ret > 0) ? static_cast<std::size_t>(ret) : 0;
    }

    void close() override {
        if (dev_) {
            if (rx_stream_) { dev_->deactivateStream(rx_stream_); dev_->closeStream(rx_stream_); }
            if (tx_stream_) { dev_->deactivateStream(tx_stream_); dev_->closeStream(tx_stream_); }
            SoapySDR::Device::unmake(dev_);
            dev_ = nullptr;
        }
    }

private:
    std::string args_;
    SoapySDR::Device* dev_ = nullptr;
    SoapySDR::Stream* rx_stream_ = nullptr;
    SoapySDR::Stream* tx_stream_ = nullptr;
};
#endif

} // namespace aegis::rt
