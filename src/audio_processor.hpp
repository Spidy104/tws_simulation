#pragma once
#include <vector>
#include <random>
#include <span>

class AudioProcessor {
public:
    // Constructor
    AudioProcessor(int filter_len = 32, float mu = 0.1f, unsigned int seed = 42);

    // Static helpers
    static std::vector<float> deinterleaveChannel(std::span<const float> interleaved_buffer,
                                                  int num_channels,
                                                  int channel_to_extract);

    static std::vector<float> getLeftChannel(std::span<const float> stereo_buffer);
    static std::vector<float> getRightChannel(std::span<const float> stereo_buffer);

    // Effects
    std::vector<float> applyPacketLoss(std::span<const float> buffer,
                                       float loss_prob,
                                       int sample_rate,
                                       int packet_ms);

    std::vector<float> applyJitter(std::span<const float> buffer,
                                   float jitter_ms,
                                   int sample_rate,
                                   int packet_ms);

    std::vector<float> applyANC(std::span<const float> primary,
                                std::span<const float> reference);

    // Allow binding helpers direct access
    friend class AudioProcessor_bindings; // optional marker

    // Public so pybind lambdas can touch them, or expose via getters/setters
    std::vector<float> filter_weights_;  // adaptive filter weights
    int filter_len_;                     // filter length
    float mu_;                           // step size
    float leak_;                         // leakage factor
    std::mt19937 rng_;                   // RNG for loss/jitter

};
