// audio_processor.cpp
#include "audio_processor.hpp"

#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <numeric>             // for std::inner_product
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>   // numpy support
#include <vector>
#include <span>

namespace py = pybind11;

// -----------------------------
// AudioProcessor methods (C++)
// -----------------------------

std::vector<float> AudioProcessor::deinterleaveChannel(
    std::span<const float> interleaved_buffer,
    int num_channels,
    int channel_to_extract)
{
    if (num_channels <= 0) throw std::invalid_argument("num_channels must be > 0");
    if (channel_to_extract < 0 || channel_to_extract >= num_channels) throw std::out_of_range("channel_to_extract");

    const std::size_t total_samples = interleaved_buffer.size();
    const std::size_t frames = total_samples / static_cast<std::size_t>(num_channels);
    std::vector<float> out;
    out.reserve(frames);

    for (std::size_t f = 0; f < frames; ++f) {
        out.push_back(interleaved_buffer[f * static_cast<std::size_t>(num_channels) + static_cast<std::size_t>(channel_to_extract)]);
    }
    return out;
}

AudioProcessor::AudioProcessor(int filter_len, float mu, unsigned int seed)
    : filter_weights_(static_cast<std::size_t>(std::max(1, filter_len)), 0.0f),
      filter_len_(std::max(1, filter_len)),
      mu_(mu),
      leak_(0.0f),
      rng_(seed)
{
    // clamp mu to a safe range for NLMS
    if (mu_ <= 0.0f) mu_ = 0.1f;
    if (mu_ > 2.0f) mu_ = 2.0f;
}

// Static helpers for stereo (2-channel) buffers
std::vector<float> AudioProcessor::getLeftChannel(std::span<const float> stereo_buffer) {
    return deinterleaveChannel(stereo_buffer, 2, 0);
}
std::vector<float> AudioProcessor::getRightChannel(std::span<const float> stereo_buffer) {
    return deinterleaveChannel(stereo_buffer, 2, 1);
}

// Packet-level loss implementation (unchanged logic, but robust checks)
std::vector<float> AudioProcessor::applyPacketLoss(std::span<const float> buffer,
                                                    float loss_prob,
                                                    int sample_rate,
                                                    int packet_ms)
{
    if (buffer.empty()) return {};
    if (sample_rate <= 0) throw std::invalid_argument("sample_rate must be positive");
    if (packet_ms <= 0) packet_ms = 20;
    if (loss_prob <= 0.0f) return std::vector<float>(buffer.begin(), buffer.end());
    if (loss_prob >= 1.0f) return std::vector<float>(buffer.size(), 0.0f);

    const int packet_size = std::max(1, (sample_rate * packet_ms) / 1000);
    const std::size_t total_samples = buffer.size();
    std::vector<float> out(total_samples, 0.0f);

    std::bernoulli_distribution drop_dist(loss_prob);

    std::size_t read_pos = 0;
    while (read_pos < total_samples) {
        std::size_t this_packet_size = std::min<std::size_t>(packet_size, total_samples - read_pos);
        bool drop = drop_dist(rng_);
        if (!drop) {
            std::copy(buffer.begin() + static_cast<std::ptrdiff_t>(read_pos),
                      buffer.begin() + static_cast<std::ptrdiff_t>(read_pos + this_packet_size),
                      out.begin() + static_cast<std::ptrdiff_t>(read_pos));
        }
        read_pos += this_packet_size;
    }
    return out;
}

// Jitter implementation (packet-level shifting)
std::vector<float> AudioProcessor::applyJitter(std::span<const float> buffer,
                                               float jitter_ms,
                                               int sample_rate,
                                               int packet_ms)
{
    if (buffer.empty()) return {};
    if (sample_rate <= 0) throw std::invalid_argument("sample_rate must be positive");
    if (packet_ms <= 0) packet_ms = 20;
    if (jitter_ms <= 0.0f) return std::vector<float>(buffer.begin(), buffer.end());

    const int packet_size = std::max(1, (sample_rate * packet_ms) / 1000);
    const int max_jitter_samples = std::lround((jitter_ms / 1000.0f) * static_cast<float>(sample_rate));

    std::uniform_int_distribution<int> jitter_dist(-max_jitter_samples, max_jitter_samples);

    const std::size_t total_samples = buffer.size();
    std::vector<float> out(total_samples, 0.0f);

    std::size_t read_pos = 0;
    while (read_pos < total_samples) {
        std::size_t this_packet_size = std::min<std::size_t>(packet_size, total_samples - read_pos);
        int delay = (max_jitter_samples == 0) ? 0 : jitter_dist(rng_);
        std::ptrdiff_t write_pos = static_cast<std::ptrdiff_t>(read_pos) + static_cast<std::ptrdiff_t>(delay);

        for (std::size_t i = 0; i < this_packet_size; ++i) {
            std::ptrdiff_t dst = write_pos + static_cast<std::ptrdiff_t>(i);
            if (dst >= 0 && static_cast<std::size_t>(dst) < total_samples) {
                out[static_cast<std::size_t>(dst)] = buffer[read_pos + i];
            }
        }
        read_pos += this_packet_size;
    }

    return out;
}

// -----------------------------
// Improved NLMS ANC (applyANC)
// - double accumulators
// - leakage term
// - weights persist by default; reset() provided
// - mu clamped to (0, 2]
// -----------------------------
std::vector<float> AudioProcessor::applyANC(std::span<const float> primary,
                                            std::span<const float> reference)
{
    const std::size_t n = std::min(primary.size(), reference.size());
    std::vector<float> out(n, 0.0f);

    if (n == 0) return out;

    const std::size_t L = static_cast<std::size_t>(filter_len_);
    if (filter_weights_.size() != L) filter_weights_.assign(L, 0.0f);

    const double eps = 1e-6; // slightly larger eps for numerical safety

    // circular buffer for last L reference samples (double precision for accuracy)
    std::vector<double> rbuf(L, 0.0);
    std::size_t rpos = 0; // index of newest sample in rbuf after insertion

    // clamp mu to stable range
    double mu = static_cast<double>(mu_);
    if (mu <= 0.0) mu = 0.1;
    if (mu > 2.0) mu = 2.0;

    double leak = static_cast<double>(leak_); // leakage factor per update (0..1), small value like 1e-4

    for (std::size_t i = 0; i < n; ++i) {
        // insert newest reference sample (as double)
        rbuf[rpos] = static_cast<double>(reference[i]);

        // compute y = w^T r_window and norm2 = ||r_window||^2 using double accumulators
        double y = 0.0;
        double norm2 = 0.0;

        std::size_t idx = rpos;
        for (std::size_t k = 0; k < L; ++k) {
            double rv = rbuf[idx];
            double wv = static_cast<double>(filter_weights_[k]);
            y += wv * rv;
            norm2 += rv * rv;
            // step backwards circularly
            idx = (idx == 0) ? (L - 1) : (idx - 1);
        }

        double e = static_cast<double>(primary[i]) - y;
        out[i] = static_cast<float>(e);

        double norm_term = norm2 + eps;
        double gain = mu / norm_term;

        // weight update with leakage
        idx = rpos;
        for (std::size_t k = 0; k < L; ++k) {
            double rv = rbuf[idx];
            double w_old = static_cast<double>(filter_weights_[k]);
            double w_new = w_old * (1.0 - leak) + gain * e * rv;
            filter_weights_[k] = static_cast<float>(w_new);
            idx = (idx == 0) ? (L - 1) : (idx - 1);
        }

        // advance circular index
        ++rpos;
        if (rpos >= L) rpos = 0;
    }

    return out;
}

// -----------------------------
// Expose internals to Python
// -----------------------------
PYBIND11_MODULE(audio_processor, m) {
    m.doc() = "AudioProcessor pybind11 bindings (numpy-friendly)";

    py::class_<AudioProcessor>(m, "AudioProcessor")
        .def(py::init<int, float, unsigned int>(),
             py::arg("filter_len") = 32,
             py::arg("mu") = 0.1f,
             py::arg("seed") = 42)

        // utility: reset adaptive filter weights to zero
        .def("reset",
             [](AudioProcessor &self) {
                 self.filter_weights_.assign(static_cast<std::size_t>(self.filter_len_), 0.0f);
             },
             "Reset adaptive filter weights to zero")

        // getters/setters for parameters
        .def_property("mu",
            [](AudioProcessor &self) { return self.mu_; },
            [](AudioProcessor &self, float v) {
                if (v <= 0.0f) v = 0.0001f;
                if (v > 2.0f) v = 2.0f;
                self.mu_ = v;
            },
            "Adaptive step size (mu). Clamped to (0, 2]."
        )

        .def_property("leak",
            [](AudioProcessor &self) { return self.leak_; },
            [](AudioProcessor &self, float v) {
                if (v < 0.0f) v = 0.0f;
                if (v > 1.0f) v = 1.0f;
                self.leak_ = v;
            },
            "Leakage factor applied to weights on each update (small, e.g. 1e-4)."
        )

        .def_property_readonly("weights",
            [](AudioProcessor &self) {
                return self.filter_weights_;
            },
            "Current adaptive filter weights (vector<float>)"
        )

        // Numpy-aware bindings: accept contiguous numpy arrays (float32)
        .def("apply_anc",
             [](AudioProcessor &self, py::array_t<float, py::array::c_style | py::array::forcecast> primary,
                               py::array_t<float, py::array::c_style | py::array::forcecast> reference) -> py::array_t<float> {
                 auto p_buf = primary.request();
                 auto r_buf = reference.request();
                 if (p_buf.ndim != 1 || r_buf.ndim != 1) throw std::runtime_error("apply_anc: expected 1-D arrays");
                 std::span<const float> p_span(static_cast<const float*>(p_buf.ptr), static_cast<std::size_t>(p_buf.size));
                 std::span<const float> r_span(static_cast<const float*>(r_buf.ptr), static_cast<std::size_t>(r_buf.size));
                 std::vector<float> out = self.applyANC(p_span, r_span);
                 // return numpy array (copy)
                 py::array_t<float> result(out.size());
                 std::memcpy(result.mutable_data(), out.data(), out.size() * sizeof(float));
                 return result;
             },
             py::arg("primary"), py::arg("reference"),
             "Apply adaptive ANC (NLMS) to primary using reference (numpy float32 arrays). Returns error signal array.")

        // accept python lists or vectors for convenience (keeps older API working)
        .def("apply_anc_list",
             [](AudioProcessor &self, const std::vector<float>& primary, const std::vector<float>& reference) -> std::vector<float> {
                 std::span<const float> p_span(primary.data(), primary.size());
                 std::span<const float> r_span(reference.data(), reference.size());
                 return self.applyANC(p_span, r_span);
             },
             py::arg("primary"), py::arg("reference"),
             "Apply ANC accepting Python lists (converts to vector)")

        // Numpy-aware packet loss
        .def("apply_packet_loss",
             [](AudioProcessor &self, py::array_t<float, py::array::c_style | py::array::forcecast> buffer,
                                float loss_prob,
                                int sample_rate,
                                int packet_ms) -> py::array_t<float> {
                 auto b = buffer.request();
                 if (b.ndim != 1) throw std::runtime_error("apply_packet_loss: expected 1-D array");
                 std::span<const float> span_buf(static_cast<const float*>(b.ptr), static_cast<std::size_t>(b.size));
                 std::vector<float> out = self.applyPacketLoss(span_buf, loss_prob, sample_rate, packet_ms);
                 py::array_t<float> res(out.size());
                 std::memcpy(res.mutable_data(), out.data(), out.size() * sizeof(float));
                 return res;
             },
             py::arg("buffer"), py::arg("loss_prob"), py::arg("sample_rate"), py::arg("packet_ms") = 20,
             "Apply packet loss to numpy float32 buffer")

        // Numpy-aware jitter
        .def("apply_jitter",
             [](AudioProcessor &self, py::array_t<float, py::array::c_style | py::array::forcecast> buffer,
                                float jitter_ms,
                                int sample_rate,
                                int packet_ms) -> py::array_t<float> {
                 auto b = buffer.request();
                 if (b.ndim != 1) throw std::runtime_error("apply_jitter: expected 1-D array");
                 std::span<const float> span_buf(static_cast<const float*>(b.ptr), static_cast<std::size_t>(b.size));
                 std::vector<float> out = self.applyJitter(span_buf, jitter_ms, sample_rate, packet_ms);
                 py::array_t<float> res(out.size());
                 std::memcpy(res.mutable_data(), out.data(), out.size() * sizeof(float));
                 return res;
             },
             py::arg("buffer"), py::arg("jitter_ms"), py::arg("sample_rate"), py::arg("packet_ms") = 20,
             "Apply packet jitter to numpy float32 buffer")

        // convenience: the original vector/list-based APIs (kept for compatibility)
        .def("apply_packet_loss_list",
             [](AudioProcessor &self, const std::vector<float>& buffer, float loss_prob, int sample_rate, int packet_ms) {
                 std::span<const float> span_buf(buffer.data(), buffer.size());
                 return self.applyPacketLoss(span_buf, loss_prob, sample_rate, packet_ms);
             })

        .def("apply_jitter_list",
             [](AudioProcessor &self, const std::vector<float>& buffer, float jitter_ms, int sample_rate, int packet_ms) {
                 std::span<const float> span_buf(buffer.data(), buffer.size());
                 return self.applyJitter(span_buf, jitter_ms, sample_rate, packet_ms);
             })
        ;
}
