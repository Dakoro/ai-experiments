#version 330 core

// Input vertex attribute (position)
layout (location = 0) in vec2 aPos;

void main()
{
    // Output the position directly; it's already in Normalized Device Coordinates
    gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
}