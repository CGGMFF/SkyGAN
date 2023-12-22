#include <iostream>

#define SKYMODEL_SPECTRUM 1
#define SKYMODEL_XYZ 2

//#define SKYMODEL_TYPE SKYMODEL_SPECTRUM
#define SKYMODEL_TYPE SKYMODEL_XYZ

#define SKYMODEL_ADD_SOLAR_DISK (SKYMODEL_TYPE != SKYMODEL_XYZ)

#if SKYMODEL_TYPE == SKYMODEL_SPECTRUM

#include "ArPragueSkyModelGroundSpectrum/ArPragueSkyModelGround.h"
#include "ArPragueSkyModelGroundSpectrum/ArPragueSkyModelGround.c"
#define SKYMODEL_DATA_FILE "ArPragueSkyModelGroundSpectrum/SkyModelDataset.dat"

#elif SKYMODEL_TYPE == SKYMODEL_XYZ

#include "ArPragueSkyModelGroundXYZ/ArPragueSkyModelGroundXYZ.h"
#include "ArPragueSkyModelGroundXYZ/ArPragueSkyModelGroundXYZ.c"
#define SKYMODEL_DATA_FILE "ArPragueSkyModelGroundXYZ/SkyModelDataset.dat"

#endif

//#include "opencv_init.cpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>

ArPragueSkyModelGroundState * state = nullptr;

double sun_elevation_ = 0;
double sun_azimuth_ = 0;
double visibility_ = 0; // in km
double albedo_ = 0; // ground albedo

void set_skymodel_params(double sun_elevation, double sun_azimuth, double visibility, double albedo) {
    if (
        sun_elevation_ == sun_elevation &&
        sun_azimuth_ == sun_azimuth &&
        visibility_ == visibility &&
        albedo_ == albedo
       ) {
        //std::cout << "noinit" << std::endl;
        return; // same as already set
    }
    
    sun_elevation_ = sun_elevation;
    sun_azimuth_ = sun_azimuth;
    visibility_ = visibility;
    albedo_ = albedo;

    if (state != nullptr) {
        //std::cout << "arpragueskymodelground_state_free" << std::endl;
        arpragueskymodelground_state_free(state);
    }

    //std::cout << "init" << std::endl;
    state = arpragueskymodelground_state_alloc_init(
        SKYMODEL_DATA_FILE,
        sun_elevation_,
        visibility_,
        albedo_
    );
    
    /*
    // fix: 0 deg (=north) should be "up"
    double sun_azimuth = 270;
    // since we are looking upwards, azimuth increases counter-clockwise, hence the minus
    sun_azimuth -= sun_azimuth_deg;
    // return to the 0..360 range
    if (sun_azimuth < 0) sun_azimuth += 360;

    sun_azimuth = sun_azimuth/180.f*M_PI; // to radians
    */
}

template<typename T=unsigned char>
struct color_template {
    T B, G, R;
    
    color_template(T R, T G, T B) : B(B), G(G), R(R) {}
};
typedef color_template<float> color;
typedef float color_element_t;

color sky_radiance(float x, float y) {
    double up_direction[] = {0.0, 0.0, 1.0};

    // stereographic projection (disc -> hemisphere)
    double view_direction[] = {0, 0, 0};
    view_direction[0] = 2*x/(1+x*x+y*y);
    view_direction[1] = 2*y/(1+x*x+y*y);
    view_direction[2] = -(-1+x*x+y*y)/(1+x*x+y*y);

    double theta, gamma, shadow;
    arpragueskymodelground_compute_angles(sun_elevation_, sun_azimuth_,
                                          view_direction, up_direction,
                                          &theta, &gamma, &shadow);

    double w1 = 1;

#if SKYMODEL_TYPE == SKYMODEL_SPECTRUM
    float r = w1 * arpragueskymodelground_sky_radiance(state, theta, gamma, shadow, 612);
    float g = w1 * arpragueskymodelground_sky_radiance(state, theta, gamma, shadow, 549);
    float b = w1 * arpragueskymodelground_sky_radiance(state, theta, gamma, shadow, 464);
#elif SKYMODEL_TYPE == SKYMODEL_XYZ
    double cie_x = arpragueskymodelground_sky_radiance(state, theta, gamma, shadow, 0);
    double cie_y = arpragueskymodelground_sky_radiance(state, theta, gamma, shadow, 1);
    double cie_z = arpragueskymodelground_sky_radiance(state, theta, gamma, shadow, 2);
    // CIE XYZ to (linear) RGB
    float r = w1 * ( 3.2404542 * cie_x - 1.5371385 * cie_y - 0.4985314 * cie_z);
    float g = w1 * (-0.9692660 * cie_x + 1.8760108 * cie_y + 0.0415560 * cie_z);
    float b = w1 * ( 0.0556434 * cie_x - 0.2040259 * cie_y + 1.0572252 * cie_z);
#endif

#if SKYMODEL_ADD_SOLAR_DISK
    // Sun angular angle (as seen from Earth) is ~0.53289 deg ... Sun radius ~695700 km, distance ~1.496e8 km
    const double solar_disk_radius_rad = 0.009300735093; // atan(695700/1.496e8)*2
    const double solar_disk_threshold = cos(solar_disk_radius_rad);

    const double sun_direction[] = {
        sin(M_PI/2 - sun_elevation_) * cos(sun_azimuth_),
        sin(M_PI/2 - sun_elevation_) * sin(sun_azimuth_),
        cos(M_PI/2 - sun_elevation_)};
    const double dot = view_direction[0] * sun_direction[0] + view_direction[1] * sun_direction[1] + view_direction[2] * sun_direction[2];
    
    if (dot >= solar_disk_threshold) {
        const double w2 = 1;
        r += w2 * arpragueskymodelground_solar_radiance(state, theta, 612);
        g += w2 * arpragueskymodelground_solar_radiance(state, theta, 549);
        b += w2 * arpragueskymodelground_solar_radiance(state, theta, 464);
        // TODO: do not *add* to the clear sky and replace instesd? ... arpragueskymodelground_solar_radiance already takes transmittance into account, but maybe not inscattered light? (that is way less than the Sun radiance so it probably does not matter...)
    }
#endif
    
    return color(r, g, b);
}

float dist_sqr(float x1, float x2, float y1, float y2) {
    return (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2);
}

void foreach_pixel_in_circle(cv::Mat & bm, std::function<void(cv::Mat & bm, int x, int y, float center_x, float center_y, float d)> f) {

    assert(bm.size.dims() == 2);
    int width = bm.cols;
    int height = bm.rows;
    float center_x = width * 0.5f;
    float center_y = height * 0.5f;

    float d = width * 0.5f;
    // TODO merge d, center_x and center_y?

    #pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (dist_sqr(center_x, x, center_y, y) < d * d) {
                f(bm, x, y, center_x, center_y, d);
            }
        }
    }
}

cv::Mat make_image(unsigned resolution, double sun_elevation, double sun_azimuth, double visibility, double albedo) {
    set_skymodel_params(sun_elevation, sun_azimuth, visibility, albedo);
    
    int width = resolution;
    int height = width;

    cv::Mat bm(width, height, CV_32FC3);

    bm = cv::Scalar(0, 0, 0);
    const auto generate_sky_color = [](cv::Mat & bm, int x, int y, float center_x, float center_y, float d) {
        float x_rel = (x - center_x) / d;
        float y_rel = (y - center_y) / d;

        color c = sky_radiance(x_rel, y_rel);
        color_element_t * p = reinterpret_cast<color_element_t *>(bm.ptr(y, x));
        *p = c.B;
        *(p+1) = c.G;
        *(p+2) = c.R;
    };
    
    foreach_pixel_in_circle(bm, generate_sky_color);
    
    return bm;
}
