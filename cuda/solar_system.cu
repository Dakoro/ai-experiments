// solar_system_cuda.cu
// CUDA-based N-body solar system simulation
// Compile with: nvcc -o solar_sim solar_system_cuda.cu -lGL -lGLU -lglut

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <GL/freeglut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PI 3.14159265359f
#define G 6.67430e-11f  // Gravitational constant
#define AU 1.496e11f    // Astronomical Unit in meters
#define TIME_SCALE 86400.0f  // Simulate days per frame

// Simulation parameters
const int NUM_BODIES = 9;  // Sun + 8 planets
const int THREADS_PER_BLOCK = 256;

// Body structure
struct Body {
    float3 pos;      // Position (m)
    float3 vel;      // Velocity (m/s)
    float mass;      // Mass (kg)
    float radius;    // Display radius (for visualization)
    float3 color;    // RGB color
    char name[20];   // Name
};

// Global variables
Body* h_bodies;           // Host bodies
Body* d_bodies;           // Device bodies
float3* d_accelerations;  // Device accelerations
float dt = TIME_SCALE;    // Time step

// Window parameters
int window_width = 1200;
int window_height = 800;
float zoom = 1.0f;
float rotX = 20.0f, rotY = 0.0f;
int lastX, lastY;
bool mouseDown = false;


static __host__ __device__
float length(const float3 &v) {
    return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
}


// Initialize solar system data
void initializeSolarSystem(Body* bodies) {
    // Sun
    bodies[0].pos = make_float3(0, 0, 0);
    bodies[0].vel = make_float3(0, 0, 0);
    bodies[0].mass = 1.989e30f;
    bodies[0].radius = 0.05f;
    bodies[0].color = make_float3(1.0f, 0.9f, 0.0f);
    strcpy(bodies[0].name, "Sun");
    
    // Mercury
    bodies[1].pos = make_float3(0.387f * AU, 0, 0);
    bodies[1].vel = make_float3(0, 47870, 0);
    bodies[1].mass = 3.301e23f;
    bodies[1].radius = 0.01f;
    bodies[1].color = make_float3(0.7f, 0.7f, 0.7f);
    strcpy(bodies[1].name, "Mercury");
    
    // Venus
    bodies[2].pos = make_float3(0.723f * AU, 0, 0);
    bodies[2].vel = make_float3(0, 35020, 0);
    bodies[2].mass = 4.867e24f;
    bodies[2].radius = 0.015f;
    bodies[2].color = make_float3(0.9f, 0.8f, 0.5f);
    strcpy(bodies[2].name, "Venus");
    
    // Earth
    bodies[3].pos = make_float3(AU, 0, 0);
    bodies[3].vel = make_float3(0, 29780, 0);
    bodies[3].mass = 5.972e24f;
    bodies[3].radius = 0.015f;
    bodies[3].color = make_float3(0.2f, 0.5f, 0.8f);
    strcpy(bodies[3].name, "Earth");
    
    // Mars
    bodies[4].pos = make_float3(1.524f * AU, 0, 0);
    bodies[4].vel = make_float3(0, 24070, 0);
    bodies[4].mass = 6.417e23f;
    bodies[4].radius = 0.012f;
    bodies[4].color = make_float3(0.8f, 0.3f, 0.2f);
    strcpy(bodies[4].name, "Mars");
    
    // Jupiter
    bodies[5].pos = make_float3(5.203f * AU, 0, 0);
    bodies[5].vel = make_float3(0, 13070, 0);
    bodies[5].mass = 1.898e27f;
    bodies[5].radius = 0.04f;
    bodies[5].color = make_float3(0.8f, 0.7f, 0.5f);
    strcpy(bodies[5].name, "Jupiter");
    
    // Saturn
    bodies[6].pos = make_float3(9.537f * AU, 0, 0);
    bodies[6].vel = make_float3(0, 9690, 0);
    bodies[6].mass = 5.683e26f;
    bodies[6].radius = 0.035f;
    bodies[6].color = make_float3(0.9f, 0.8f, 0.6f);
    strcpy(bodies[6].name, "Saturn");
    
    // Uranus
    bodies[7].pos = make_float3(19.191f * AU, 0, 0);
    bodies[7].vel = make_float3(0, 6810, 0);
    bodies[7].mass = 8.681e25f;
    bodies[7].radius = 0.025f;
    bodies[7].color = make_float3(0.5f, 0.8f, 0.9f);
    strcpy(bodies[7].name, "Uranus");
    
    // Neptune
    bodies[8].pos = make_float3(30.069f * AU, 0, 0);
    bodies[8].vel = make_float3(0, 5430, 0);
    bodies[8].mass = 1.024e26f;
    bodies[8].radius = 0.025f;
    bodies[8].color = make_float3(0.3f, 0.5f, 0.9f);
    strcpy(bodies[8].name, "Neptune");
}

// CUDA kernel to compute gravitational accelerations
__global__ void computeAccelerations(Body* bodies, float3* accelerations, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float3 acc = make_float3(0, 0, 0);
    float3 pos_i = bodies[i].pos;
    
    // Calculate gravitational force from all other bodies
    for (int j = 0; j < n; j++) {
        if (i != j) {
            float3 pos_j = bodies[j].pos;
            float3 r = make_float3(pos_j.x - pos_i.x, 
                                  pos_j.y - pos_i.y, 
                                  pos_j.z - pos_i.z);
            
            float dist2 = r.x * r.x + r.y * r.y + r.z * r.z;
            float dist = sqrtf(dist2);
            
            // Avoid singularity
            if (dist > 1e6f) {
                float force = G * bodies[j].mass / dist2;
                float inv_dist = 1.0f / dist;
                
                acc.x += force * r.x * inv_dist;
                acc.y += force * r.y * inv_dist;
                acc.z += force * r.z * inv_dist;
            }
        }
    }
    
    accelerations[i] = acc;
}

// CUDA kernel to update positions and velocities
__global__ void updateBodies(Body* bodies, float3* accelerations, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    // Update velocity
    bodies[i].vel.x += accelerations[i].x * dt;
    bodies[i].vel.y += accelerations[i].y * dt;
    bodies[i].vel.z += accelerations[i].z * dt;
    
    // Update position
    bodies[i].pos.x += bodies[i].vel.x * dt;
    bodies[i].pos.y += bodies[i].vel.y * dt;
    bodies[i].pos.z += bodies[i].vel.z * dt;
}

// Simulate one time step
void simulateStep() {
    int blocks = (NUM_BODIES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // Compute accelerations
    computeAccelerations<<<blocks, THREADS_PER_BLOCK>>>(d_bodies, d_accelerations, NUM_BODIES);
    cudaDeviceSynchronize();
    
    // Update positions and velocities
    updateBodies<<<blocks, THREADS_PER_BLOCK>>>(d_bodies, d_accelerations, dt, NUM_BODIES);
    cudaDeviceSynchronize();
    
    // Copy back to host
    cudaMemcpy(h_bodies, d_bodies, NUM_BODIES * sizeof(Body), cudaMemcpyDeviceToHost);
}

// Draw a sphere
void drawSphere(float radius, int slices, int stacks) {
    GLUquadric* quadric = gluNewQuadric();
    gluSphere(quadric, radius, slices, stacks);
    gluDeleteQuadric(quadric);
}

// Draw orbit trail
void drawOrbitTrail(int body_index, float scale) {
    glBegin(GL_LINE_LOOP);
    glColor3f(0.3f, 0.3f, 0.3f);
    
    // Simple circular approximation for visualization
    float r = length(h_bodies[body_index].pos) / AU * scale;
    for (int i = 0; i < 100; i++) {
        float angle = 2.0f * PI * i / 100.0f;
        glVertex3f(r * cos(angle), 0, r * sin(angle));
    }
    glEnd();
}

// OpenGL display function
void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    // Camera position
    gluLookAt(0, 5 * zoom, 10 * zoom,  // Eye position
              0, 0, 0,                  // Look at
              0, 1, 0);                 // Up vector
    
    // Apply rotations
    glRotatef(rotX, 1, 0, 0);
    glRotatef(rotY, 0, 1, 0);
    
    float scale = 3.0f / AU;  // Scale for visualization
    
    // Draw orbit trails
    for (int i = 1; i < NUM_BODIES; i++) {
        drawOrbitTrail(i, scale);
    }
    
    // Draw bodies
    for (int i = 0; i < NUM_BODIES; i++) {
        glPushMatrix();
        
        // Scale and translate
        float3 pos = h_bodies[i].pos;
        glTranslatef(pos.x * scale, pos.y * scale, pos.z * scale);
        
        // Set color
        glColor3f(h_bodies[i].color.x, h_bodies[i].color.y, h_bodies[i].color.z);
        
        // Draw sphere
        float display_radius = h_bodies[i].radius;
        if (i == 0) display_radius *= 2.0f;  // Make sun bigger for visibility
        drawSphere(display_radius, 20, 20);
        
        // Draw label
        glRasterPos3f(0, display_radius + 0.02f, 0);
        for (char* c = h_bodies[i].name; *c != '\0'; c++) {
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, *c);
        }
        
        glPopMatrix();
    }
    
    // Display info
    char info[100];
    glColor3f(1, 1, 1);
    // With this block:
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, window_width, 0, window_height, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    // Now (10, window_height - 20) is in window‐pixel coords:
    glRasterPos2i(10, window_height - 20);
    sprintf(info, "Solar System Simulation - Time scale: %.0f days/frame", dt / 86400.0f);
    for (char* c = info; *c; c++)
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *c);

    glPopMatrix();               // modelview
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    
    glutSwapBuffers();
}

// OpenGL idle function
void idle() {
    simulateStep();
    glutPostRedisplay();
}

// Keyboard handler
void keyboard(unsigned char key, int x, int y) {
    switch (key) {
        case 27:  // ESC
            // Cleanup
            cudaFree(d_bodies);
            cudaFree(d_accelerations);
            free(h_bodies);
            exit(0);
            break;
        case '+':
            zoom *= 0.9f;
            break;
        case '-':
            zoom *= 1.1f;
            break;
        case ' ':
            // Toggle pause (not implemented in this simple version)
            break;
        case 'r':
            // Reset
            initializeSolarSystem(h_bodies);
            cudaMemcpy(d_bodies, h_bodies, NUM_BODIES * sizeof(Body), cudaMemcpyHostToDevice);
            break;
    }
}

// Mouse handlers
void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
        if (state == GLUT_DOWN) {
            mouseDown = true;
            lastX = x;
            lastY = y;
        } else {
            mouseDown = false;
        }
    }
    
    // Mouse wheel zoom
    if (button == 3) zoom *= 0.9f;  // Scroll up
    if (button == 4) zoom *= 1.1f;  // Scroll down
}

void motion(int x, int y) {
    if (mouseDown) {
        rotY += (x - lastX) * 0.5f;
        rotX += (y - lastY) * 0.5f;
        lastX = x;
        lastY = y;
    }
}

// OpenGL reshape function
void reshape(int w, int h) {
    window_width = w;
    window_height = h;
    glViewport(0, 0, w, h);
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, (double)w/h, 0.1, 100.0);
}

// Initialize OpenGL
void initGL() {
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);
    
    GLfloat light_position[] = {0.0, 0.0, 0.0, 1.0};
    GLfloat light_ambient[] = {0.2, 0.2, 0.2, 1.0};
    GLfloat light_diffuse[] = {1.0, 1.0, 1.0, 1.0};
    
    glLightfv(GL_LIGHT0, GL_POSITION, light_position);
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
    
    glClearColor(0.0, 0.0, 0.05, 1.0);
}

int main(int argc, char** argv) {
    // Initialize GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("CUDA Solar System Simulation");
    
    // Initialize OpenGL
    initGL();
    
    // Allocate host memory
    h_bodies = (Body*)malloc(NUM_BODIES * sizeof(Body));
    initializeSolarSystem(h_bodies);
    
    // Allocate device memory
    cudaMalloc(&d_bodies, NUM_BODIES * sizeof(Body));
    cudaMalloc(&d_accelerations, NUM_BODIES * sizeof(float3));
    
    // Copy initial data to device
    cudaMemcpy(d_bodies, h_bodies, NUM_BODIES * sizeof(Body), cudaMemcpyHostToDevice);
    
    // Set up callbacks
    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutReshapeFunc(reshape);
    
    // Print controls
    printf("Solar System CUDA Simulation\n");
    printf("Controls:\n");
    printf("  Mouse: Rotate view\n");
    printf("  +/-: Zoom in/out\n");
    printf("  r: Reset simulation\n");
    printf("  ESC: Exit\n");
    
    // Start main loop
    glutMainLoop();
    
    return 0;
}