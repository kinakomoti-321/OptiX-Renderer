#pragma once
#include <cuda_runtime.h>
#include <sutil/vec_math.h>
#include <myPathTracer/math.h>
#include <myPathTracer/constant.h>
#include <myPathTracer/random.h>

static __device__ float3 shlick_fresnel(float3 F0,const float in) {
    float delta = fmaxf(1.0f - in, 0.0f);
    return F0 + (1.0f - F0) * delta * delta * delta * delta * delta;
}

class GGX {

private:
    float3 F0;
    float alpha;
    
    __device__ float lambda(const float3& v) const {
        float delta = fmaxf(alpha * BSDFMath::tanTheta(v), 0.0f);
        return fmaxf((-1.0f + sqrtf(1.0f + delta * delta)) / 2.0f, 0.0f);
    }
    
    //Height correlated Smith shadowing-masking
    __device__ float shadowG(const float3& o,const float3& i) {
        return 1.0f / (1.0f + lambda(o) + lambda(i));
    }
    __device__ float shadowG_1(const float3& v) {
        return 1.0f / (1.0f + lambda(v));
    }

    //GGX normal distiribution
    __device__ float GGX_D(const float3& m) {
        const float tan2theta = BSDFMath::tan2Theta(m);
        const float cos4theta = BSDFMath::cos2Theta(m) * BSDFMath::cos2Theta(m);
        const float term = 1.0f + tan2theta / (alpha * alpha);
        return 1.0f / ((PI * alpha * alpha * cos4theta) * term * term);
    }
    
    //Importance Sampling
    //Walter distribution sampling
    __device__ float3 walterSampling(float u, float v) {
        float theta = atanf(alpha * sqrtf(v) / sqrtf(fmaxf(1.0f - v, 0.0f)));
        float phi = 2.0f * PI * u;
        return hemisphereVector(theta,phi);
    }

    __device__ float3 visibleNormalSampling(const float3& V_,float u,float v) {
        float a_x = alpha, a_y = alpha;
        float3 V = normalize(make_float3(a_x * V_.x, V_.y, a_y * V_.z));

        float3 n = make_float3(0, 1, 0);
        if (V.y > 0.99) n = make_float3(1, 0, 0);
        float3 T1 = normalize(cross(V, n));
        float3 T2 = normalize(cross(T1, V));

        float r = sqrtf(u);
        float a = 1.0f / (1.0f + V.y);
        float phi;
        if (a > v) {
            phi = PI * v / a;
        }
        else {
            phi = PI * (v - a) / (1.0f - a) + PI;
        }

        float P1 = r * cosf(phi);
        float P2 = r * sinf(phi);
        if (a < v) P2 *= V.y;

        float3 N = P1 * T1 + P2 * T2 + sqrtf(fmaxf(1.0f - P1 * P1 - P2 * P2, 0.0f)) * V;

        N = normalize(make_float3(a_x * N.x, N.y, a_y * N.z));
        return N;
    }

public:
    __device__ GGX(){
        F0 = { 0.0,0.0,0.0 };
        alpha = 0.0f;
    }
    __device__ GGX(const float3& F0,const float& in_alpha):F0(F0){
        alpha = fmaxf(in_alpha, 0.01f);
        alpha = alpha * alpha;

    }

    __device__ float3 sampleBSDF(const float3& wo, float3& wi, float& pdf, unsigned int& seed) {
        float3 i = wo;
        float3 n = { 0.0, 1.0, 0.0 };

        //Walter Sampling
        //float3 m = walterSampling(rnd(seed), rnd(seed));

        //Visible Normal Sampling
        float3 m = visibleNormalSampling(i,rnd(seed), rnd(seed));

        float3 o = reflect(-wo, m);
        wi = o;
        //if (wi.y < 0.0f) return { 0.0,0.0,0.0 };

        float im = absDot(i, m);
        float in = absDot(i, n);
        float on = absDot(o, n);

        float3 F = shlick_fresnel(F0, im);
        float G_ = shadowG(o, i);
        float D_ = GGX_D(m);

        float3 brdf = F * G_ * D_ / (4.0f * in * on);

        //Walter sampling PDF
        //pdf = D_ * BSDFMath::cosTheta(m) / (4.0f * absDot(m, o));

        //Visible Normal Sampling PDF
        pdf = D_ * shadowG_1(i) * im / (absDot(i, n)*4.0f * absDot(m,o));

        return brdf;
    }

    __device__ float3 evaluateBSDF(const float3& wo, float3& wi) {
        float3 i = wo;
        float3 n = { 0.0, 1.0, 0.0 };
        float3 m = normalize(wi + wo);
        float3 o = wi;
        wi = o;
        if (wi.y < 0.0f) return { 0.0,0.0,0.0 };

        float im = absDot(i, m);
        float in = absDot(i, n);
        float on = absDot(o, n);

        float3 F = shlick_fresnel(F0, im);
        float G_ = shadowG(o, i);
        float D_ = GGX_D(m);

        float3 brdf = F * G_ * D_ / (4.0f * in * on);

        return brdf;
    }

    __device__ float pdfBSDF(const float3& wo,const float3& wi) {
        float3 i = wo;
        float3 m = normalize(wi + wo);
        float3 o = wi;
        float3 n = make_float3(0, 1, 0);
        float im = absDot(i, m);

        float D_ = GGX_D(m);
        return D_ * BSDFMath::cosTheta(m) / (4.0f * absDot(m, o)) * shadowG_1(i) * im / absDot(i, n);
    }
};

class IdealReflect {
private: 
    float3 rho;
public:
    __device__ IdealReflect() {
        rho = { 0.0,0.0,0.0 };
    }

    __device__ IdealReflect(const float3& rho):rho(rho) {}
    __device__ float3 sampleBSDF(const float3& wo, float3& wi, float& pdf, unsigned int& seed) {
        float3 n = { 0.0,1.0,0.0 };
        wi = reflect(-wo, n);
        pdf = 1.0f;
        return rho;
    }

};
