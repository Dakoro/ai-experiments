#version 330 core

out vec4 FragColor;

// Uniforms passed from C
uniform vec2 u_resolution;
uniform float u_time;

// --- Ray Marching Constants ---
const int MAX_MARCH_STEPS = 256;
const float MAX_DIST = 100.0;
const float SURF_DIST = 0.001;
const float POWER = 8.0;

/**
 * @brief Power-8 Mandelbulb Distance Estimator (DE).
 * This function calculates the shortest distance from point `p` to the
 * surface of the Mandelbulb set.
 * Based on the formula by Daniel White and Paul Nylander.
 */
float mandelbulbDE(vec3 p) {
    vec3 z = p;
    float dr = 1.0;  // Derivative
    float r = length(z);
    int iterations = 0;

    for (int i = 0; i < 20; i++) { // Max iterations for DE calculation
        if (r > 4.0) break; // Bailout condition
        
        // Convert to spherical coordinates
        float theta = acos(z.y / r);
        float phi = atan(z.x, z.z); // Note: atan(x, z) gives correct angle

        // Update derivative: dr = |POWER * r^(POWER-1) * dr + 1|
        dr = pow(r, POWER - 1.0) * POWER * dr + 1.0;

        // Mandelbulb iteration: z = r^POWER * vec3(s(theta*P)*c(phi*P), c(theta*P), s(theta*P)*s(phi*P)) + p
        float zr = pow(r, POWER);
        theta = theta * POWER;
        phi = phi * POWER;
        
        z = zr * vec3(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi));
        z += p; // Add original point (Mandelbrot set style)
        
        r = length(z);
        iterations = i;
    }
    
    // Return distance estimation
    return 0.5 * log(r) * r / dr;
}

/**
 * @brief Calculates the surface normal at point `p` using finite differences.
 * This is done by sampling the DE at slightly offset points.
 */
vec3 getNormal(vec3 p) {
    vec2 e = vec2(SURF_DIST, 0.0);
    return normalize(vec3(
        mandelbulbDE(p + e.xyy) - mandelbulbDE(p - e.xyy),
        mandelbulbDE(p + e.yxy) - mandelbulbDE(p - e.yxy),
        mandelbulbDE(p + e.yyx) - mandelbulbDE(p - e.yyx)
    ));
}

/**
 * @brief Performs the ray marching loop.
 * Starts at ray origin `ro` and steps along ray direction `rd`.
 * Returns the total distance traveled, or MAX_DIST if no surface is hit.
 */
float rayMarch(vec3 ro, vec3 rd) {
    float dO = 0.0; // Total distance traveled
    for (int i = 0; i < MAX_MARCH_STEPS; i++) {
        vec3 p = ro + rd * dO; // Current point along the ray
        float dS = mandelbulbDE(p); // Distance from p to surface
        
        dO += dS; // Step forward by the estimated distance
        
        // Hit condition or max distance check
        if (dS < SURF_DIST || dO > MAX_DIST) break;
    }
    return dO;
}

/**
 * @brief Converts HSV (Hue, Saturation, Value) color to RGB.
 */
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

/**
 * @brief Creates a rotation matrix for rotating around the Y-axis by angle `a`.
 */
mat3 rotY(float a) {
    float s = sin(a);
    float c = cos(a);
    return mat3(c, 0, -s, 0, 1, 0, s, 0, c);
}

// --- Main Shader Function ---
void main() {
    // 1. Calculate UV coordinates
    // This maps pixel coordinates [0, res] to centered, aspect-corrected [-asp, asp]x[-1, 1]
    vec2 uv = (2.0 * gl_FragCoord.xy - u_resolution.xy) / u_resolution.y;
    
    // 2. Setup Scene
    vec3 ro = vec3(0.0, 0.0, 3.5); // Ray Origin (camera position)
    vec3 rd = normalize(vec3(uv, -1.5)); // Ray Direction (camera "lens")
    vec3 lightPos = vec3(2.0, 3.0, 4.0); // Light position
    
    // 3. Apply Rotation
    // Rotate camera and light around Y-axis based on time
    float angle = u_time * 0.25; // Control rotation speed
    mat3 rot = rotY(angle);
    ro = rot * ro;
    rd = rot * rd;
    lightPos = rot * lightPos;
    
    // 4. Ray March
    float d = rayMarch(ro, rd);
    
    // 5. Shading and Coloring
    if (d < MAX_DIST) {
        // We hit the surface
        vec3 p = ro + rd * d; // Hit position
        vec3 n = getNormal(p); // Surface normal
        
        // Lighting
        vec3 l = normalize(lightPos - p); // Light direction
        float dif = clamp(dot(n, l), 0.0, 1.0); // Diffuse
        float amb = 0.2; // Ambient
        
        // Dynamic Coloring
        // Use position and time to create a "trippy" psychedelic effect
        float hue = mod(length(p) * 0.1 - u_time * 0.1, 1.0);
        vec3 col = hsv2rgb(vec3(hue, 0.8, 1.0)); // Bright, saturated color
        
        // Final color is lit color
        FragColor = vec4(col * (dif + amb), 1.0);
    } else {
        // We missed (hit the "sky")
        FragColor = vec4(0.0, 0.0, 0.0, 1.0); // Black background
    }
}