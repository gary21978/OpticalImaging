#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "farfield_hopkins.h"

class FarFieldImager
{
  public:
    FarFieldImager() = default;
    ~FarFieldImager() { ClearKernelCache(); }

    FarFieldImager(HopkinsSourceConfig source,
                   HopkinsProjectorConfig projector = {},
                   HopkinsNumericsConfig numerics = {})
        : source_(std::move(source)),
          projector_(std::move(projector)),
          numerics_(std::move(numerics))
    {
    }

    HopkinsSourceConfig& source_config() { return source_; }
    const HopkinsSourceConfig& source_config() const { return source_; }

    HopkinsProjectorConfig& projector_config() { return projector_; }
    const HopkinsProjectorConfig& projector_config() const { return projector_; }

    HopkinsNumericsConfig& numerics_config() { return numerics_; }
    const HopkinsNumericsConfig& numerics_config() const { return numerics_; }

    HopkinsImageResult& image() { return image_; }
    const HopkinsImageResult& image() const { return image_; }

    HopkinsKernelCache& kernel_cache() { return kernel_cache_; }
    const HopkinsKernelCache& kernel_cache() const { return kernel_cache_; }

    bool HasKernelCache() const { return kernel_cache_.valid; }

    void ClearImage() { image_ = HopkinsImageResult{}; }
    void ClearKernelCache()
    {
        ReleaseHopkinsKernelCache(&kernel_cache_);
        kernel_cache_.Clear();
    }

    Status ComputeFromFields(const Complex* ex_field,
                             const Complex* ey_field,
                             const Complex* ez_field,
                             int64_t width,
                             int64_t height,
                             Real period_x,
                             Real period_y,
                             cudaStream_t stream = nullptr)
    {
        return CalculateHopkinsImageFromFields(ex_field,
                                               ey_field,
                                               ez_field,
                                               width,
                                               height,
                                               period_x,
                                               period_y,
                                               source_,
                                               projector_,
                                               numerics_,
                                               &image_,
                                               &kernel_cache_,
                                               stream);
    }

    Status ComputeFromRCWA(const RCWA& rcwa,
                           const Real* x_axis,
                           int64_t x_count,
                           const Real* y_axis,
                           int64_t y_count,
                           Real period_x,
                           Real period_y,
                           cudaStream_t stream = nullptr)
    {
        return CalculateHopkinsImageFromRCWA(rcwa,
                                             x_axis,
                                             x_count,
                                             y_axis,
                                             y_count,
                                             period_x,
                                             period_y,
                                             source_,
                                             projector_,
                                             numerics_,
                                             &image_,
                                             &kernel_cache_,
                                             stream);
    }

    std::vector<Real> GetFocusSlice(int64_t focus_index) const
    {
        if (focus_index < 0 || focus_index >= image_.focus_count || image_.width <= 0 || image_.height <= 0)
        {
            return {};
        }

        const size_t slice_size = static_cast<size_t>(image_.width) * static_cast<size_t>(image_.height);
        const size_t offset = static_cast<size_t>(focus_index) * slice_size;
        return std::vector<Real>(image_.intensity.begin() + offset,
                                 image_.intensity.begin() + offset + slice_size);
    }

  private:
    HopkinsSourceConfig source_{};
    HopkinsProjectorConfig projector_{};
    HopkinsNumericsConfig numerics_{};
    HopkinsImageResult image_{};
    HopkinsKernelCache kernel_cache_{};
};
