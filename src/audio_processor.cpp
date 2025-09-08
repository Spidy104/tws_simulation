// audio_processor.cpp
#include "audio_processor.hpp"

#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <span>

namespace py = pybind11;

// Constants for improved numerical stability and validation
namespace {
    constexpr double DEFAULT_EPSILON = 1e-8;
    constexpr float MAX_MU = 2.0f;
    constexpr float MIN_MU = 1e-6f;
    constexpr float MAX_LEAKAGE = 1.0f;
    constexpr int MAX_FILTER_LENGTH = 8192;
    constexpr int MIN_FILTER_LENGTH = 1;
}

// -----------------------------
// AudioProcessor methods (C++)
// -----------------------------

std::vector<float> AudioProcessor::deinterleaveChannel(
    std::span<const float> interleaved_buffer,
    int num_channels,
    int channel_to_extract)
{
    if (num_channels <= 0) {
        throw std::invalid_argument("Number of channels must be positive");
    }
    if (channel_to_extract < 0 || channel_to_extract >= num_channels) {
        throw std::out_of_range("Channel index out of range");
    }
    if (interleaved_buffer.size() % static_cast<size_t>(num_channels) != 0) {
        throw std::invalid_argument("Buffer size not divisible by number of channels");
    }

    const size_t total_samples = interleaved_buffer.size();
    const size_t frames = total_samples / static_cast<size_t>(num_channels);
    
    std::vector<float> out;
    out.reserve(frames);

    for (size_t frame = 0; frame < frames; ++frame) {
        const size_t index = frame * static_cast<size_t>(num_channels) + 
                            static_cast<size_t>(channel_to_extract);
        out.push_back(interleaved_buffer[index]);
    }
    
    return out;
}

AudioProcessor::AudioProcessor(int filter_len, float mu, unsigned int seed)
    : filter_weights_(static_cast<size_t>(std::clamp(filter_len, MIN_FILTER_LENGTH, MAX_FILTER_LENGTH)), 0.0f),
      filter_len_(std::clamp(filter_len, MIN_FILTER_LENGTH, MAX_FILTER_LENGTH)),
      mu_(std::clamp(mu, MIN_MU, MAX_MU)),
      leak_(0.0f),
      rng_(seed)
{
    // Validation ensures parameters are within safe ranges
    if (filter_len_ <= 0) filter_len_ = 1;
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

// Enhanced packet-level loss implementation with better error handling
std::vector<float> AudioProcessor::applyPacketLoss(std::span<const float> buffer,
                                                    float loss_prob,
                                                    int sample_rate,
                                                    int packet_ms)
{
    if (buffer.empty()) return {};
    if (sample_rate <= 0) {
        throw std::invalid_argument("Sample rate must be positive");
    }
    
    // Clamp loss probability to valid range
    loss_prob = std::clamp(loss_prob, 0.0f, 1.0f);
    packet_ms = std::max(1, packet_ms);
    
    // Early returns for edge cases
    if (loss_prob <= 0.0f) {
        return std::vector<float>(buffer.begin(), buffer.end());
    }
    if (loss_prob >= 1.0f) {
        return std::vector<float>(buffer.size(), 0.0f);
    }

    const int packet_size = std::max(1, (sample_rate * packet_ms) / 1000);
    const size_t total_samples = buffer.size();
    std::vector<float> out;
    out.reserve(total_samples);
    out.resize(total_samples, 0.0f);

    std::bernoulli_distribution drop_dist(static_cast<double>(loss_prob));

    for (size_t read_pos = 0; read_pos < total_samples; read_pos += static_cast<size_t>(packet_size)) {
        const size_t this_packet_size = std::min<size_t>(packet_size, total_samples - read_pos);
        
        if (!drop_dist(rng_)) {
            // Copy packet if not dropped
            std::copy(buffer.begin() + static_cast<ptrdiff_t>(read_pos),
                      buffer.begin() + static_cast<ptrdiff_t>(read_pos + this_packet_size),
                      out.begin() + static_cast<ptrdiff_t>(read_pos));
        }
        // If dropped, leave zeros (already initialized)
    }
    
    return out;
}

// Enhanced jitter implementation with improved bounds checking
std::vector<float> AudioProcessor::applyJitter(std::span<const float> buffer,
                                               float jitter_ms,
                                               int sample_rate,
                                               int packet_ms)
{
    if (buffer.empty()) return {};
    if (sample_rate <= 0) {
        throw std::invalid_argument("Sample rate must be positive");
    }
    
    jitter_ms = std::max(0.0f, jitter_ms);
    packet_ms = std::max(1, packet_ms);
    
    // Early return if no jitter
    if (jitter_ms <= 0.0f) {
        return std::vector<float>(buffer.begin(), buffer.end());
    }

    const int packet_size = std::max(1, (sample_rate * packet_ms) / 1000);
    const int max_jitter_samples = static_cast<int>((jitter_ms / 1000.0f) * static_cast<float>(sample_rate));
    
    if (max_jitter_samples == 0) {
        return std::vector<float>(buffer.begin(), buffer.end());
    }

    const size_t total_samples = buffer.size();
    // Create larger buffer to accommodate jitter
    const size_t extended_size = total_samples + 2 * static_cast<size_t>(max_jitter_samples);
    std::vector<float> extended_out(extended_size, 0.0f);

    std::uniform_int_distribution<int> jitter_dist(-max_jitter_samples, max_jitter_samples);

    // Process each packet with random jitter
    for (size_t read_pos = 0; read_pos < total_samples; read_pos += static_cast<size_t>(packet_size)) {
        const size_t this_packet_size = std::min<size_t>(packet_size, total_samples - read_pos);
        const int delay = jitter_dist(rng_);
        const ptrdiff_t write_pos = static_cast<ptrdiff_t>(read_pos + static_cast<size_t>(max_jitter_samples) + static_cast<size_t>(delay));

        // Bounds checking for write position
        if (write_pos >= 0 && static_cast<size_t>(write_pos) + this_packet_size <= extended_size) {
            std::copy(buffer.begin() + static_cast<ptrdiff_t>(read_pos),
                     buffer.begin() + static_cast<ptrdiff_t>(read_pos + this_packet_size),
                     extended_out.begin() + write_pos);
        }
    }

    // Extract the original-sized portion from the center
    const size_t start_idx = static_cast<size_t>(max_jitter_samples);
    const size_t end_idx = std::min(start_idx + total_samples, extended_size);
    
    return std::vector<float>(extended_out.begin() + static_cast<ptrdiff_t>(start_idx),
                             extended_out.begin() + static_cast<ptrdiff_t>(end_idx));
}

// -----------------------------
// Significantly improved NLMS ANC (applyANC)
// - Uses double precision for all intermediate calculations
// - Circular buffer for efficient reference sample storage
// - Better numerical stability with larger epsilon
// - Proper weight normalization and leakage
// -----------------------------
std::vector<float> AudioProcessor::applyANC(std::span<const float> primary,
                                            std::span<const float> reference)
{
    const size_t n = std::min(primary.size(), reference.size());
    if (n == 0) return {};

    std::vector<float> out;
    out.reserve(n);

    const size_t L = static_cast<size_t>(filter_len_);
    
    // Ensure filter weights are properly sized
    if (filter_weights_.size() != L) {
        filter_weights_.assign(L, 0.0f);
    }

    // Use circular buffer for reference samples (double precision for accuracy)
    std::vector<double> reference_circular(L, 0.0);
    size_t circular_idx = 0;

    // Clamp mu to numerically stable range
    const double mu_d = std::clamp(static_cast<double>(mu_), static_cast<double>(MIN_MU), static_cast<double>(MAX_MU));
    const double leak_d = std::clamp(static_cast<double>(leak_), 0.0, static_cast<double>(MAX_LEAKAGE));
    const double leakage_factor = 1.0 - leak_d;

    for (size_t i = 0; i < n; ++i) {
        // Add new reference sample to circular buffer
        reference_circular[circular_idx] = static_cast<double>(reference[i]);

        // Compute filter output and reference power using double precision
        double filter_output = 0.0;
        double reference_power = 0.0;
        
        // Walk backwards through circular buffer to get time-reversed order
        size_t buf_idx = circular_idx;
        for (size_t k = 0; k < L; ++k) {
            const double ref_val = reference_circular[buf_idx];
            const double weight_val = static_cast<double>(filter_weights_[k]);
            
            filter_output += weight_val * ref_val;
            reference_power += ref_val * ref_val;
            
            // Move backwards in circular buffer
            buf_idx = (buf_idx == 0) ? (L - 1) : (buf_idx - 1);
        }

        // Calculate error signal
        const double error = static_cast<double>(primary[i]) - filter_output;
        out.push_back(static_cast<float>(error));

        // NLMS weight update with improved numerical stability
        const double norm_factor = reference_power + DEFAULT_EPSILON;
        const double step_size = mu_d / norm_factor;

        // Update weights with leakage
        buf_idx = circular_idx;
        for (size_t k = 0; k < L; ++k) {
            const double ref_val = reference_circular[buf_idx];
            const double old_weight = static_cast<double>(filter_weights_[k]);
            
            // Apply leakage and adaptation
            const double new_weight = old_weight * leakage_factor + step_size * error * ref_val;
            filter_weights_[k] = static_cast<float>(new_weight);
            
            buf_idx = (buf_idx == 0) ? (L - 1) : (buf_idx - 1);
        }

        // Advance circular buffer index
        circular_idx = (circular_idx + 1) % L;
    }

    return out;
}

// -----------------------------
// Enhanced Python bindings with improved error handling
// -----------------------------
PYBIND11_MODULE(audio_processor, m) {
    m.doc() = "Enhanced AudioProcessor with improved performance and numerical stability";

    py::class_<AudioProcessor>(m, "AudioProcessor")
        .def(py::init<int, float, unsigned int>(),
             py::arg("filter_len") = 32,
             py::arg("mu") = 0.1f,
             py::arg("seed") = 42,
             "Create AudioProcessor with specified parameters")

        // Utility: reset adaptive filter weights to zero
        .def("reset",
             [](AudioProcessor &self) {
                 self.filter_weights_.assign(static_cast<size_t>(self.filter_len_), 0.0f);
             },
             "Reset adaptive filter weights to zero")

        // Enhanced getters/setters with validation
        .def_property("mu",
            [](AudioProcessor &self) { return self.mu_; },
            [](AudioProcessor &self, float v) {
                self.mu_ = std::clamp(v, MIN_MU, MAX_MU);
            },
            "Adaptive step size (mu). Automatically clamped to valid range [1e-6, 2.0]."
        )

        .def_property("leak",
            [](AudioProcessor &self) { return self.leak_; },
            [](AudioProcessor &self, float v) {
                self.leak_ = std::clamp(v, 0.0f, MAX_LEAKAGE);
            },
            "Leakage factor applied to weights on each update. Range [0.0, 1.0]."
        )

        .def_property("filter_length",
            [](AudioProcessor &self) { return self.filter_len_; },
            [](AudioProcessor &self, int v) {
                self.filter_len_ = std::clamp(v, MIN_FILTER_LENGTH, MAX_FILTER_LENGTH);
                self.filter_weights_.assign(static_cast<size_t>(self.filter_len_), 0.0f);
            },
            "Adaptive filter length. Resets weights when changed."
        )

        .def_property_readonly("weights",
            [](AudioProcessor &self) {
                return self.filter_weights_;
            },
            "Current adaptive filter weights (read-only)"
        )

        // Enhanced numpy-aware bindings with better error handling
        .def("apply_anc",
             [](AudioProcessor &self, 
                py::array_t<float, py::array::c_style | py::array::forcecast> primary,
                py::array_t<float, py::array::c_style | py::array::forcecast> reference) -> py::array_t<float> {
                 
                 auto p_info = primary.request();
                 auto r_info = reference.request();
                 
                 if (p_info.ndim != 1 || r_info.ndim != 1) {
                     throw std::runtime_error("Input arrays must be 1-dimensional");
                 }
                 if (p_info.size == 0 || r_info.size == 0) {
                     throw std::runtime_error("Input arrays cannot be empty");
                 }
                 
                 std::span<const float> p_span(static_cast<const float*>(p_info.ptr), 
                                             static_cast<size_t>(p_info.size));
                 std::span<const float> r_span(static_cast<const float*>(r_info.ptr), 
                                             static_cast<size_t>(r_info.size));
                 
                 auto result = self.applyANC(p_span, r_span);
                 
                 py::array_t<float> output(result.size());
                 if (!result.empty()) {
                     std::memcpy(output.mutable_data(), result.data(), result.size() * sizeof(float));
                 }
                 return output;
             },
             py::arg("primary"), py::arg("reference"),
             "Apply adaptive noise cancellation using NLMS algorithm")

        // Backward compatibility: accept python lists/vectors
        .def("apply_anc_list",
             [](AudioProcessor &self, const std::vector<float>& primary, const std::vector<float>& reference) -> std::vector<float> {
                 if (primary.empty() || reference.empty()) {
                     throw std::runtime_error("Input vectors cannot be empty");
                 }
                 std::span<const float> p_span(primary.data(), primary.size());
                 std::span<const float> r_span(reference.data(), reference.size());
                 return self.applyANC(p_span, r_span);
             },
             py::arg("primary"), py::arg("reference"),
             "Apply ANC accepting Python lists (legacy compatibility)")

        // Enhanced numpy-aware packet loss
        .def("apply_packet_loss",
             [](AudioProcessor &self, 
                py::array_t<float, py::array::c_style | py::array::forcecast> buffer,
                float loss_prob,
                int sample_rate,
                int packet_ms) -> py::array_t<float> {
                 
                 auto buf_info = buffer.request();
                 if (buf_info.ndim != 1) {
                     throw std::runtime_error("Buffer must be 1-dimensional");
                 }
                 if (buf_info.size == 0) {
                     return py::array_t<float>(0);
                 }
                 
                 std::span<const float> buf_span(static_cast<const float*>(buf_info.ptr),
                                                static_cast<size_t>(buf_info.size));
                 
                 auto result = self.applyPacketLoss(buf_span, loss_prob, sample_rate, packet_ms);
                 
                 py::array_t<float> output(result.size());
                 if (!result.empty()) {
                     std::memcpy(output.mutable_data(), result.data(), result.size() * sizeof(float));
                 }
                 return output;
             },
             py::arg("buffer"), py::arg("loss_prob"), py::arg("sample_rate") = 44100, py::arg("packet_ms") = 20,
             "Apply packet loss simulation to numpy array")

        // Enhanced numpy-aware jitter
        .def("apply_jitter",
             [](AudioProcessor &self,
                py::array_t<float, py::array::c_style | py::array::forcecast> buffer,
                float jitter_ms,
                int sample_rate,
                int packet_ms) -> py::array_t<float> {
                 
                 auto buf_info = buffer.request();
                 if (buf_info.ndim != 1) {
                     throw std::runtime_error("Buffer must be 1-dimensional");
                 }
                 if (buf_info.size == 0) {
                     return py::array_t<float>(0);
                 }
                 
                 std::span<const float> buf_span(static_cast<const float*>(buf_info.ptr),
                                                static_cast<size_t>(buf_info.size));
                 
                 auto result = self.applyJitter(buf_span, jitter_ms, sample_rate, packet_ms);
                 
                 py::array_t<float> output(result.size());
                 if (!result.empty()) {
                     std::memcpy(output.mutable_data(), result.data(), result.size() * sizeof(float));
                 }
                 return output;
             },
             py::arg("buffer"), py::arg("jitter_ms"), py::arg("sample_rate") = 44100, py::arg("packet_ms") = 20,
             "Apply packet jitter simulation to numpy array")

        // Legacy list-based APIs for backward compatibility
        .def("apply_packet_loss_list",
             [](AudioProcessor &self, const std::vector<float>& buffer, float loss_prob, int sample_rate, int packet_ms) {
                 if (buffer.empty()) return std::vector<float>{};
                 std::span<const float> span_buf(buffer.data(), buffer.size());
                 return self.applyPacketLoss(span_buf, loss_prob, sample_rate, packet_ms);
             },
             py::arg("buffer"), py::arg("loss_prob"), py::arg("sample_rate"), py::arg("packet_ms") = 20,
             "Apply packet loss to Python list (legacy compatibility)")

        .def("apply_jitter_list",
             [](AudioProcessor &self, const std::vector<float>& buffer, float jitter_ms, int sample_rate, int packet_ms) {
                 if (buffer.empty()) return std::vector<float>{};
                 std::span<const float> span_buf(buffer.data(), buffer.size());
                 return self.applyJitter(span_buf, jitter_ms, sample_rate, packet_ms);
             },
             py::arg("buffer"), py::arg("jitter_ms"), py::arg("sample_rate"), py::arg("packet_ms") = 20,
             "Apply jitter to Python list (legacy compatibility)")
        ;

    // Static utility functions exposed at module level
    m.def("deinterleave_channel", &AudioProcessor::deinterleaveChannel,
          py::arg("buffer"), py::arg("num_channels"), py::arg("channel_index"),
          "Extract a single channel from interleaved audio data");
    
    m.def("get_left_channel", [](py::array_t<float, py::array::c_style | py::array::forcecast> buffer) -> py::array_t<float> {
        auto buf_info = buffer.request();
        if (buf_info.ndim != 1) {
            throw std::runtime_error("Buffer must be 1-dimensional");
        }
        if (buf_info.size == 0) {
            return py::array_t<float>(0);
        }
        
        std::span<const float> buf_span(static_cast<const float*>(buf_info.ptr),
                                       static_cast<size_t>(buf_info.size));
        auto result = AudioProcessor::getLeftChannel(buf_span);
        
        py::array_t<float> output(result.size());
        if (!result.empty()) {
            std::memcpy(output.mutable_data(), result.data(), result.size() * sizeof(float));
        }
        return output;
    }, py::arg("stereo_buffer"), "Extract left channel from stereo numpy array");
    
    m.def("get_right_channel", [](py::array_t<float, py::array::c_style | py::array::forcecast> buffer) -> py::array_t<float> {
        auto buf_info = buffer.request();
        if (buf_info.ndim != 1) {
            throw std::runtime_error("Buffer must be 1-dimensional");
        }
        if (buf_info.size == 0) {
            return py::array_t<float>(0);
        }
        
        std::span<const float> buf_span(static_cast<const float*>(buf_info.ptr),
                                       static_cast<size_t>(buf_info.size));
        auto result = AudioProcessor::getRightChannel(buf_span);
        
        py::array_t<float> output(result.size());
        if (!result.empty()) {
            std::memcpy(output.mutable_data(), result.data(), result.size() * sizeof(float));
        }
        return output;
    }, py::arg("stereo_buffer"), "Extract right channel from stereo numpy array");
}
