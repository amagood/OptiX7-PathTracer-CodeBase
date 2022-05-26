// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <optix_device.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "myString.cuh"

#include "LaunchParams.h"
#include "gdt/random/random.h"
#include "TriangleData.cuh"

//#define PARALLEL_LIGHT

using namespace osc;

#define NUM_LIGHT_SAMPLES 1
#define NUM_PIXEL_SAMPLES 8
constexpr int RRBeginDepth = 4;
#define maxDepth 7

__device__ vec3f missColor;

namespace osc
{

    typedef gdt::LCG<16> Random;

    /*! launch parameters in constant memory, filled in by optix upon
        optixLaunch (this gets filled in from the buffer we pass to
        optixLaunch) */
    extern "C" __constant__ LaunchParams optixLaunchParams;

    /*! per-ray data now captures random number generator, so programs
        can access RNG state */
    struct PRD
    {
        Random random;
        vec3f pixelColor;
        vec3f nextRayOrigin;
        vec3f nextRayDirection;
        vec3i ch_triangle_index; //only updated when hit glass for now
        int depth;
        char pathREGEX[maxDepth] = {};

        bool isEnd;
        bool anyHitLight;
    };

    static __forceinline__ __device__
    void *unpackPointer(uint32_t i0, uint32_t i1)
    {
        const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
        void *ptr = reinterpret_cast<void *>( uptr );
        return ptr;
    }

    static __forceinline__ __device__
    void packPointer(void *ptr, uint32_t &i0, uint32_t &i1)
    {
        const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
        i0 = uptr >> 32;
        i1 = uptr & 0x00000000ffffffff;
    }

    template<typename T>
    static __forceinline__ __device__ T *getPRD()
    {
        const uint32_t u0 = optixGetPayload_0();
        const uint32_t u1 = optixGetPayload_1();
        return reinterpret_cast<T *>( unpackPointer(u0, u1));
    }

    //------------------------------------------------------------------------------
    // closest hit and anyhit programs for radiance-type rays.
    //
    // Note eventually we will have to create one pair of those for each
    // ray type and each geometry type we want to render; but this
    // simple example doesn't use any actual geometries yet, so we only
    // create a single, dummy, set of them (we do have to have at least
    // one group of them to set up the SBT)
    //------------------------------------------------------------------------------

    extern "C" __global__ void __closesthit__shadow()
    {
        /* not going to be used ... */
    }

    struct Onb
    {
        __forceinline__ __device__ Onb(const vec3f &normal)
        {
            m_tangent = vec3f(0.f);
            m_binormal = vec3f(0.f);
            m_normal = vec3f(0.f);

            m_normal = normal;

            if (fabs(m_normal.x) > fabs(m_normal.z))
            {
                m_binormal.x = -m_normal.y;
                m_binormal.y = m_normal.x;
                m_binormal.z = 0;
            }
            else
            {
                m_binormal.x = 0;
                m_binormal.y = -m_normal.z;
                m_binormal.z = m_normal.y;
            }

            m_binormal = normalize(m_binormal);
            m_tangent = cross(m_binormal, m_normal);
        }

        __forceinline__ __device__ void inverse_transform(vec3f &p) const
        {
            p = p.x * m_tangent + p.y * m_binormal + p.z * m_normal;
        }

        vec3f m_tangent;
        vec3f m_binormal;
        vec3f m_normal;
    };

static __device__ __inline__ vec3f reflect(vec3f rayDir, vec3f normal)
{
    return rayDir - 2.f * dot(normal, rayDir) * normal;
}

static __device__ __inline__ vec3f refract(vec3f rayDir, vec3f normal, float eta)
{
    const float k = 1.f - eta * eta * (1.f - dot(normal, rayDir) * dot(normal, rayDir));
    if(k < 0.f)
        return vec3f(0.f);
    else
        return eta * rayDir - (eta * dot(normal, rayDir) + sqrt(k)) * normal;
}

static __device__ __inline__ bool refract(vec3f &w_t, vec3f rayDir, vec3f normal, float eta)
{
    const float k = 1.f - eta * eta * (1.f - dot(normal, rayDir) * dot(normal, rayDir));
    if(k < 0.f)
    {
        w_t = vec3f(0.f);
        return false;
    }
    else
    {
        w_t = eta * rayDir - (eta * dot(normal, rayDir) + sqrt(k)) * normal;
        return true;
    }
}

static __device__ __inline__ float fresnel(float cos_theta_i, float cos_theta_t, float eta)
{
    const float rs = (cos_theta_i - cos_theta_t * eta) / (cos_theta_i + eta * cos_theta_t);
    const float rp = (cos_theta_i * eta - cos_theta_t) / (cos_theta_i * eta + cos_theta_t);

    return 0.5f * (rs * rs + rp * rp);
}

static __forceinline__ __device__ void cosine_sample_hemisphere(const float u1, const float u2, vec3f& p)
{
    // Uniformly sample disk.
    const float r   = sqrtf( u1 );
    const float phi = 2.0f * 3.141592653589793 * u2;
    p.x = r * cosf( phi );
    p.y = r * sinf( phi );

    // Project up to hemisphere.
    p.z = sqrtf( fmaxf( 0.0f, 1.0f - p.x*p.x - p.y*p.y ) );
}


extern "C" __global__ void __anyhit__radiance() { /*! for this simple example, this will remain empty */ }

extern "C" __global__ void __anyhit__shadow() { /*! not going to be used */ }

//------------------------------------------------------------------------------
// miss program that gets called for any ray that did not have a
// valid intersection
//
// as with the anyhit/closest hit programs, in this example we only
// need to have _some_ dummy function to set up a valid SBT
// ------------------------------------------------------------------------------

extern "C" __global__ void __miss__radiance()
{
    PRD &prd = *getPRD<PRD>();
    // set to constant white as background color
    prd.isEnd = true;
    prd.pixelColor *= missColor;
}

extern "C" __global__ void __miss__shadow()
{
    // we didn't hit anything, so the light is visible
    vec3f &prd = *(vec3f *) getPRD<vec3f>();
    prd = vec3f(0.f);
}

__device__ bool init = false;
//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------
extern "C" __global__ void __raygen__renderFrame()
{
    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
    const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x; //frame buffer index

    PPD &nowPPD = optixLaunchParams.frame.ppdBuffer[fbIndex]; //get the per pixel data of this pixel


    const int accumID = optixLaunchParams.frame.accumID;
    const auto &camera = optixLaunchParams.camera;

    if(accumID == 0 && !init)
    {
        init = true;
        missColor = vec3f(0.f);
    }

    PRD prd;
    prd.random.init(ix + accumID * optixLaunchParams.frame.size.x,
                    iy + accumID * optixLaunchParams.frame.size.y);

    // the values we store the PRD pointer in:
    uint32_t u0, u1;
    packPointer(&prd, u0, u1);

    int numPixelSamples = NUM_PIXEL_SAMPLES;

    vec3f pixelColor = 0.f;
    for (int sampleID = 0; sampleID < numPixelSamples; sampleID++)
    {
        //clear prd
        prd.isEnd = false;
        prd.depth = 0;
        prd.nextRayDirection = vec3f(0.f);
        prd.nextRayOrigin = vec3f(0.f);
        my_strcpy(prd.pathREGEX, "");

        // normalized screen plane position, in [0,1]^2
        const vec2f screen(vec2f(ix + prd.random(), iy + prd.random())
                           / vec2f(optixLaunchParams.frame.size));

        // generate ray direction
        vec3f rayDir = normalize(camera.direction
                                 + (screen.x - 0.5f) * camera.horizontal
                                 + (screen.y - 0.5f) * camera.vertical);
        vec3f rayOrigin = camera.position;
        prd.pixelColor = vec3f(1.f);

        for (int i = 0; !prd.isEnd; i++)
        {
            optixTrace(optixLaunchParams.traversable,
                       rayOrigin,
                       rayDir,
                       0.0001f,    // tmin
                       1e20f,  // tmax
                       0.0f,   // rayTime
                       OptixVisibilityMask(255),
                       OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
                       RADIANCE_RAY_TYPE,            // SBT offset
                       RAY_TYPE_COUNT,               // SBT stride
                       RADIANCE_RAY_TYPE,            // missSBTIndex
                       u0, u1);

            if(prd.depth >= RRBeginDepth)
            {
                float p = length(prd.pixelColor);
                p = min(p , 1.f);
                if(prd.random() >= p)
                    break;
                prd.pixelColor /= p;
            }
            if(prd.depth >= maxDepth)
                break;

            if(prd.isEnd)
                break;

            rayOrigin = prd.nextRayOrigin;
            rayDir = prd.nextRayDirection;
        }
        if(prd.isEnd)
        {
            pixelColor += prd.pixelColor;
        }
    }
    // and write to frame buffer ...
    if (accumID == 0)
    {
        const int r = int(255.99f * min(pixelColor.x / numPixelSamples, 1.f));
        const int g = int(255.99f * min(pixelColor.y / numPixelSamples, 1.f));
        const int b = int(255.99f * min(pixelColor.z / numPixelSamples, 1.f));

        // convert to 32-bit rgba value (we explicitly set alpha to 0xff
        // to make stb_image_write happy ...
        const uint32_t rgba = 0xff000000 | (r << 0) | (g << 8) | (b << 16);
        optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
        optixLaunchParams.frame.accumulateBuffer[fbIndex] = pixelColor / numPixelSamples;
    }
    else
    {
        vec3f prevColor = optixLaunchParams.frame.accumulateBuffer[fbIndex];
        vec3f newColor = prevColor + (((pixelColor / numPixelSamples) - prevColor) / (accumID + 1));

        optixLaunchParams.frame.accumulateBuffer[fbIndex] = newColor;
        const int r = int(255.99f * min(newColor.x, 1.f));
        const int g = int(255.99f * min(newColor.y, 1.f));
        const int b = int(255.99f * min(newColor.z, 1.f));

        // convert to 32-bit rgba value (we explicitly set alpha to 0xff
        // to make stb_image_write happy ...
        const uint32_t rgba = 0xff000000 | (r << 0) | (g << 8) | (b << 16);
        optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
    }
}

extern "C" __global__ void __closesthit__radiance()  //diffuse
{
    const TriangleMeshSBTData &sbtData
            = *(const TriangleMeshSBTData *) optixGetSbtDataPointer();
    PRD &prd = *getPRD<PRD>();
    prd.depth++;

    const TriangleData triangleData(sbtData);

    // start with some ambient term
    vec3f pixelColor = triangleData.diffuseColor;

    const float z1 = prd.random();
    const float z2 = prd.random();
    vec3f w_in = normalize(vec3f(1, 1, 1));
    cosine_sample_hemisphere( z1, z2, w_in );
    Onb onb(triangleData.Ns);
    onb.inverse_transform( w_in );

    prd.nextRayOrigin = triangleData.surfPos + 1e-3f * triangleData.Ns;
    prd.nextRayDirection = w_in;
    prd.pixelColor *= pixelColor;
    my_strcat(prd.pathREGEX, "D");
}

extern "C" __global__ void __closesthit__metal()
{
    const TriangleMeshSBTData &sbtData
            = *(const TriangleMeshSBTData *) optixGetSbtDataPointer();
    PRD &prd = *getPRD<PRD>();

    const TriangleData triangleData(sbtData);

    vec3f reflectDirection = reflect(triangleData.rayDir, triangleData.Ns);

    prd.nextRayOrigin = triangleData.surfPos + 1e-3f * triangleData.Ns;
    prd.nextRayDirection = reflectDirection;
    prd.depth++;
    my_strcat(prd.pathREGEX, "M");
}

extern "C" __global__ void __closesthit__glass()
{
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
    const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x; //frame buffer index

    PPD &nowPPD = optixLaunchParams.frame.ppdBuffer[fbIndex]; //get the per pixel data of this pixel

    const TriangleMeshSBTData &sbtData
            = *(const TriangleMeshSBTData *) optixGetSbtDataPointer();
    PRD &prd = *getPRD<PRD>();

    const TriangleData triangleData(sbtData);


    // ------------------------------------------------------------------
    // face-forward and normalize normals
    // ------------------------------------------------------------------
    vec3f frontFacedNormal = triangleData.rawNormal;
    float wavelengthIor = sbtData.refractionIndex;

    float cos_theta_i = dot(-triangleData.rayDir, frontFacedNormal);
    float eta;
    float t_hit = optixGetRayTmax();

    vec3f extinction(-1 * log(0.9f), -1 * log(0.9f), -1 * log(0.9f));
    vec3f transmittance(1.f);
    if(cos_theta_i > 0.f)
    {
        // Ray is entering
        eta = wavelengthIor;// Note: does not handle nested dielectrics
    }
    else
    {
        // Ray is exiting; apply Beer's Law.
        // This is derived in Shirley's Fundamentals of Graphics book.
        transmittance = vec3f(expf(-extinction.x * t_hit), expf(-extinction.y * t_hit), expf(-extinction.z * t_hit));

        eta = 1.f / wavelengthIor;
        cos_theta_i = -cos_theta_i;
        frontFacedNormal = -frontFacedNormal;
    }

    vec3f w_t;
    const bool tir = !refract(w_t, triangleData.rayDir, frontFacedNormal, eta);
    const float cos_theta_t = -dot(frontFacedNormal, w_t);
    float R = tir ? 1.f : fresnel(cos_theta_i, cos_theta_t, eta);

    const float z = prd.random();
    if(z <= R)
    {
        //Reflect
        const vec3f w_in = reflect(normalize(triangleData.rayDir), normalize(frontFacedNormal));
        prd.nextRayDirection = w_in;
        my_strcat(prd.pathREGEX, "R");
    }
    else
    {
        //Refract
        const vec3f w_in = w_t;
        prd.nextRayDirection = w_in;
        my_strcat(prd.pathREGEX, "S");
    }
    prd.nextRayOrigin = triangleData.surfPos;

    prd.pixelColor *= transmittance;
    prd.depth++;
}

extern "C" __global__ void __closesthit__light()
{
    const TriangleMeshSBTData &sbtData
            = *(const TriangleMeshSBTData *) optixGetSbtDataPointer();
    PRD &prd = *getPRD<PRD>();

    const TriangleData triangleData(sbtData);

#ifndef PARALLEL_LIGHT //area light
    prd.pixelColor *= sbtData.emissionColor;
    prd.depth++;
    prd.isEnd = true;
    my_strcat(prd.pathREGEX, "L");
#else //parallel light

    const vec3f lightSourceDir = -triangleData.Ns;  //the normal of the light plate

    if(dot(triangleData.rayDir, lightSourceDir) > 0.99f)  //not perfect parallel is ok
    {
        prd.pixelColor *= sbtData.emissionColor;
        prd.depth++;
        prd.isEnd = true;
        my_strcat(prd.pathREGEX, "L");
    }
    else //not parallel is considered miss
    {
        prd.isEnd = true;
        prd.depth++;
        prd.pixelColor *= missColor;
    }

#endif
}

extern "C" __global__ void __anyhit__light()
{
    const TriangleMeshSBTData &sbtData
            = *(const TriangleMeshSBTData *) optixGetSbtDataPointer();
    PRD &prd = *getPRD<PRD>();

    const TriangleData triangleData(sbtData);

    const vec3f lightSourceDir = -triangleData.Ns;  //the normal of the light plate

#ifndef PARALLEL_LIGHT //area light
    prd.anyHitLight = true;
#else //parallel light
    if(dot(triangleData.rayDir, lightSourceDir) > 0.99f) prd.anyHitLight = true;
#endif
}

} // ::osc
