// -*- mode: c++ -*-

//**************
// Visualization
//**************

struct ray{
  float4 origin;
  float4 direction;
};

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

int intersectBox(float4 r_o, float4 r_d, float4 boxmin, float4 boxmax, float *tnear, float *tfar)
{
  // compute intersection of ray with all six bbox planes
  float4 invR = (float4)(1.0f,1.0f,1.0f,1.0f) / r_d;
  float4 tbot = invR * (boxmin - r_o);
  float4 ttop = invR * (boxmax - r_o);

  // re-order intersections to find smallest and largest on each axis
  float4 tmin = min(ttop, tbot);
  float4 tmax = max(ttop, tbot);

  // find the largest tmin and the smallest tmax
  float largest_tmin = max(max(tmin.x, tmin.y), max(tmin.x, tmin.z));
  float smallest_tmax = min(min(tmax.x, tmax.y), min(tmax.x, tmax.z));

  *tnear = largest_tmin;
  *tfar = smallest_tmax;

  return smallest_tmax > largest_tmin;
}

float getDensityFromVolume(const float4 p, const int resolution, __global float* volumeData){
  int x = p.x * resolution;
  int y = p.y * resolution;
  int z = p.z * resolution;

  if(x > resolution -1 || x < 0) return(0.0f);
  if(y > resolution -1 || y < 0) return(0.0f);
  if(z > resolution -1 || z < 0) return(0.0f);

  return volumeData[x + y * resolution + z * resolution * resolution];
}

float4 getNormalFromVolume(const float4 p, const int resolution, __global float* volumeData){
  float4 normal;

  normal.x = getDensityFromVolume((float4)(p.x + 2.0f/resolution, p.y, p.z, 0.0f), resolution, volumeData) -
    getDensityFromVolume((float4)(p.x - 2.0f/resolution, p.y, p.z, 0.0f), resolution, volumeData);
  normal.y = getDensityFromVolume((float4)(p.x, p.y + 2.0f/resolution, p.z, 0.0f), resolution, volumeData) -
    getDensityFromVolume((float4)(p.x, p.y - 2.0f/resolution, p.z, 0.0f), resolution, volumeData);
  normal.z = getDensityFromVolume((float4)(p.x, p.y, p.z + 2.0f/resolution, 0.0f), resolution, volumeData) -
    getDensityFromVolume((float4)(p.x, p.y, p.z - 2.0f/resolution, 0.0f), resolution, volumeData);
  normal.w = 0.0f;

  if(dot(normal, normal) < 0.001f){
    normal = (float4)(0.0f, 0.0f, 1.0f, 0.0f);
  }

  return normalize(normal);
}

// Iso-sourface raycasting
__kernel
void isosurface(const int width, const int height, __global float4* visualizationBuffer,
                   const int resolution, __global float* volumeData,
		   const float isoValue, const float16 invViewMatrix){
  int2 id = (int2)(get_global_id(0), get_global_id(1));

  float2 uv = (float2)( (id.x / (float) width)*2.0f-1.0f, (id.y / (float) height)*2.0f-1.0f );

  float4 boxMin = (float4)(-1.0f, -1.0f, -1.0f,1.0f);
  float4 boxMax = (float4)(1.0f, 1.0f, 1.0f,1.0f);

  // calculate eye ray in world space
  struct ray eyeRay;

  eyeRay.origin = (float4)(invViewMatrix.sC, invViewMatrix.sD, invViewMatrix.sE, invViewMatrix.sF);

  float4 temp = normalize(((float4)(uv.x, uv.y, -2.0f, 0.0f)));
  eyeRay.direction.x = dot(temp, ((float4)(invViewMatrix.s0,invViewMatrix.s1,invViewMatrix.s2,invViewMatrix.s3)));
  eyeRay.direction.y = dot(temp, ((float4)(invViewMatrix.s4,invViewMatrix.s5,invViewMatrix.s6,invViewMatrix.s7)));
  eyeRay.direction.z = dot(temp, ((float4)(invViewMatrix.s8,invViewMatrix.s9,invViewMatrix.sA,invViewMatrix.sB)));
  eyeRay.direction.w = 0.0f;

  float4 color = (float4)(0.0f);
  float4 lightDir = (float4)(0.3f, -2.0f, 0.0f, 0.0f);

  //printf("%f %f %f\n", eyeRay.direction.x, eyeRay.direction.y, eyeRay.direction.z);

  float tnear, tfar;
  int hit = intersectBox(eyeRay.origin, eyeRay.direction, boxMin, boxMax, &tnear, &tfar);
  if(hit){
      
      float lastDensity = getDensityFromVolume(eyeRay.origin, resolution, volumeData);
      float4 step = eyeRay.direction / resolution;
      float4 next = eyeRay.origin + step;
      int found = 0;
      while ((next.z < 1.0) && !found) {
          float nextDensity = getDensityFromVolume(next, resolution, volumeData);
          if (lastDensity < nextDensity) {
              if (lastDensity < isoValue && isoValue < nextDensity) {
                  found = 1;
              }
          }
          else {
              if (nextDensity < isoValue && isoValue < lastDensity) {
                  found = 1;
              }
          }
          next = next + step;
          lastDensity = nextDensity;
      }
      /* 1A
      if (found) {
          color = (float4)(1.0f);
      }
      */
      /* 1B
      if (found) {
          float4 normal = getNormalFromVolume(next, resolution, volumeData);
          color = (float4)clamp(dot(normal, lightDir), 0.0f, 1);
      }
      */
      //1C
      if (found) {
          float4 normal = getNormalFromVolume(next, resolution, volumeData);
          color = (float4)0.5f + 0.5f * dot(normal, lightDir);
      }
  }

  if(id.x < width && id.y < height){
    visualizationBuffer[id.x + id.y * width] = color;
  }
}

// Alpha blended
__kernel
void alphaBlended(const int width, const int height, __global float4* visualizationBuffer,
		  const int resolution, __global float* volumeData,
		  const float alphaExponent, const float alphaCenter,
		  const float16 invViewMatrix){
  int2 id = (int2)(get_global_id(0), get_global_id(1));

  float2 uv = (float2)( (id.x / (float) width)*2.0f-1.0f, (id.y / (float) height)*2.0f-1.0f );

  float4 boxMin = (float4)(-1.0f, -1.0f, -1.0f,1.0f);
  float4 boxMax = (float4)(1.0f, 1.0f, 1.0f,1.0f);

  // calculate eye ray in world space
  struct ray eyeRay;

  eyeRay.origin = (float4)(invViewMatrix.sC, invViewMatrix.sD, invViewMatrix.sE, invViewMatrix.sF);

  float4 temp = normalize(((float4)(uv.x, uv.y, -2.0f, 0.0f)));
  eyeRay.direction.x = dot(temp, ((float4)(invViewMatrix.s0,invViewMatrix.s1,invViewMatrix.s2,invViewMatrix.s3)));
  eyeRay.direction.y = dot(temp, ((float4)(invViewMatrix.s4,invViewMatrix.s5,invViewMatrix.s6,invViewMatrix.s7)));
  eyeRay.direction.z = dot(temp, ((float4)(invViewMatrix.s8,invViewMatrix.s9,invViewMatrix.sA,invViewMatrix.sB)));
  eyeRay.direction.w = 0.0f;

  float4 sum = (float4)(0.0f);

  float tnear, tfar;
  int hit = intersectBox(eyeRay.origin, eyeRay.direction, boxMin, boxMax, &tnear, &tfar);
  if(hit){
      float4 step = eyeRay.direction / resolution;
      float4 current = eyeRay.origin + eyeRay.direction * tfar;
      float4 end = eyeRay.origin + eyeRay.direction * tnear;
      float e = 3.0f / resolution;
      
      float dd = 0.0f;
      do {
          float density = getDensityFromVolume(current, resolution, volumeData);
          float alpha = pow(density, alphaExponent) / resolution;
          sum = (1 - alpha) * sum + alpha * (float4)(1.0);

          float4 d = end - current;
          dd = sqrt(d.x * d.x + d.z * d.z + d.y * d.y);
          current -= step;
      } while (dd < e);
  }

  if(id.x < width && id.y < height){
    visualizationBuffer[id.x + id.y * width] = (float4)(sum);
  }
}

