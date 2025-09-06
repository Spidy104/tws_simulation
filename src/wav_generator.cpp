// src/wav_generator.cpp
#include "wav_generator.hpp"

#include <sndfile.h>
#include <cmath>
#include <stdexcept>
#include <memory>
#include <cstddef>        // for std::size_t
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>      // <<< add this: enables std::vector/std::string conversions
#include <pybind11/stl_bind.h> // optional; not required for simple types
#include <pybind11/complex.h>  // only if you use complex types

namespace py = pybind11;

// Portable definition for pi (don't rely on C++20 std::numbers here)
static constexpr double PI = 3.14159265358979323846;

// Constructor (matches header: noexcept)
WavGenerator::WavGenerator(int sample_rate, double duration_s) noexcept
    : sample_rate(sample_rate), duration_s(duration_s) {}

// Private helper method to handle file writing
void WavGenerator::writeWavFile(const std::string& filename, const std::vector<float>& buffer) const {
    SF_INFO sfinfo{};
    sfinfo.samplerate = this->sample_rate;
    sfinfo.channels = 2;
    // frames = samples per channel
    sfinfo.frames = static_cast<sf_count_t>(buffer.size() / static_cast<std::size_t>(sfinfo.channels));
    sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

    std::unique_ptr<SNDFILE, decltype(&sf_close)> file(
        sf_open(filename.c_str(), SFM_WRITE, &sfinfo),
        &sf_close
    );
    if (!file) {
        throw std::runtime_error("Failed to open WAV file for writing: " + filename);
    }

    sf_count_t items_written = sf_write_float(file.get(), buffer.data(), static_cast<sf_count_t>(buffer.size()));
    if (items_written != static_cast<sf_count_t>(buffer.size())) {
        throw std::runtime_error("Error writing to WAV file: " + filename);
    }
}

void WavGenerator::generateStereoWav(const std::string& filename,
                                     double freq_left,
                                     double freq_right,
                                     double amplitude) const {
    const auto num_samples = static_cast<long long>(static_cast<long long>(sample_rate) * duration_s);
    std::vector<float> buffer(static_cast<std::size_t>(num_samples) * 2u);

    const double angular_freq_left  = 2.0 * PI * freq_left / static_cast<double>(sample_rate);
    const double angular_freq_right = 2.0 * PI * freq_right / static_cast<double>(sample_rate);

    for (long long i = 0; i < num_samples; ++i) {
        const double t = static_cast<double>(i);
        buffer[2 * static_cast<std::size_t>(i)]     = static_cast<float>(amplitude * std::sin(angular_freq_left * t));
        buffer[2 * static_cast<std::size_t>(i) + 1] = static_cast<float>(amplitude * std::sin(angular_freq_right * t));
    }

    writeWavFile(filename, buffer);
}

void WavGenerator::generateANCWav(const std::string& filename,
                                  double signal_freq,
                                  double noise_freq,
                                  double signal_amp,
                                  double noise_amp) const {
    const auto num_samples = static_cast<long long>(static_cast<long long>(sample_rate) * duration_s);
    std::vector<float> buffer(static_cast<std::size_t>(num_samples) * 2u);

    const double angular_freq_signal = 2.0 * PI * signal_freq / static_cast<double>(sample_rate);
    const double angular_freq_noise  = 2.0 * PI * noise_freq / static_cast<double>(sample_rate);

    for (long long i = 0; i < num_samples; ++i) {
        const double t = static_cast<double>(i);
        float signal = static_cast<float>(signal_amp * std::sin(angular_freq_signal * t));
        float noise  = static_cast<float>(noise_amp  * std::sin(angular_freq_noise  * t));

        buffer[2 * static_cast<std::size_t>(i)]     = signal + noise; // Left = primary
        buffer[2 * static_cast<std::size_t>(i) + 1] = noise;          // Right = reference
    }

    writeWavFile(filename, buffer);
}

WavData WavGenerator::readWav(const std::string& filename) const {
    SF_INFO sfinfo{};
    std::unique_ptr<SNDFILE, decltype(&sf_close)> file(
        sf_open(filename.c_str(), SFM_READ, &sfinfo),
        &sf_close
    );
    if (!file) {
        throw std::runtime_error("Failed to open WAV file for reading: " + filename);
    }

    WavData result;
    result.sample_rate = sfinfo.samplerate;
    result.channels = sfinfo.channels;
    result.samples.resize(static_cast<std::size_t>(sfinfo.frames) * static_cast<std::size_t>(sfinfo.channels));

    sf_count_t items_read = sf_read_float(file.get(), result.samples.data(), static_cast<sf_count_t>(result.samples.size()));
    if (items_read != static_cast<sf_count_t>(result.samples.size())) {
         throw std::runtime_error("Error reading from WAV file: " + filename);
    }

    return result;
}

// pybind11 bindings
PYBIND11_MODULE(wav_generator, m) {
    py::class_<WavData>(m, "WavData")
        .def(py::init<>())
        .def_readwrite("samples", &WavData::samples)
        .def_readwrite("channels", &WavData::channels)
        .def_readwrite("sample_rate", &WavData::sample_rate);

    py::class_<WavGenerator>(m, "WavGenerator")
        .def(py::init<int, double>(),
             py::arg("sample_rate") = 44100,
             py::arg("duration_s") = 5.0)
        .def("generate_stereo_wav", &WavGenerator::generateStereoWav,
             py::arg("filename"), py::arg("freq_left"),
             py::arg("freq_right"), py::arg("amplitude") = 0.8)
        .def("generate_anc_wav", &WavGenerator::generateANCWav,
             py::arg("filename"),
             py::arg("signal_freq") = 440.0,
             py::arg("noise_freq")  = 200.0,
             py::arg("signal_amp")  = 0.8,
             py::arg("noise_amp")   = 0.3)
        .def("read_wav", &WavGenerator::readWav,
             py::arg("filename"));
}
