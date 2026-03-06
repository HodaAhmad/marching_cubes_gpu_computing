// File: main.cpp
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define MAX_TRIANGLES 64*1024*1024

glm::vec3 cameraPos   = glm::vec3(0.0f, 5.0f, 0.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, -0.3f, -1.0f);  // Looking down
glm::vec3 cameraUp    = glm::vec3(0.0f, 1.0f, 0.0f);

float yaw   = -90.0f;
float pitch = 0.0f;
float lastX = 400.0f, lastY = 300.0f;
bool firstMouse = true;

float cameraSpeed = 2.5f;
float sensitivity = 0.1f;

struct float4 {
    float x, y, z, w;
};
static_assert(sizeof(float4) == 16, "float4 size mismatch");


GLuint createShader(GLenum type, const char* path) {
    std::ifstream file(path);
    std::stringstream ss;
    ss << file.rdbuf();
    std::string source = ss.str();
    const char* src = source.c_str();

    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(shader, 512, nullptr, log);
        std::cerr << "Shader compile error (" << path << "):\n" << log << std::endl;
    }

    return shader;
}

GLuint createProgram(const char* vertPath, const char* fragPath, const char* geomPath = nullptr) {
    GLuint vs = createShader(GL_VERTEX_SHADER, vertPath);
    GLuint fs = createShader(GL_FRAGMENT_SHADER, fragPath);
    GLuint gs = createShader(GL_GEOMETRY_SHADER, geomPath);
    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, gs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    return program;
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;

    xoffset *= sensitivity;
    yoffset *= sensitivity;

    yaw   += xoffset;
    pitch += yoffset;

    if (pitch > 89.0f) pitch = 89.0f;
    if (pitch < -89.0f) pitch = -89.0f;

    glm::vec3 direction;
    direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    direction.y = sin(glm::radians(pitch));
    direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraFront = glm::normalize(direction);
}

void processInput(GLFWwindow* window, float deltaTime) {
    float velocity = cameraSpeed * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cameraPos += velocity * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraPos -= velocity * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * velocity;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * velocity;

    // Escape key to close the window
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
}

void readTrianglesFromFile(const char *filename, std::vector<float4> &vertices,
                      std::vector<unsigned int> &indices)
{
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error opening file %s for reading\n", filename);
        return;
    }

    size_t r;

    unsigned int vertexCount, indexCount;

    // Read vertex count and index count
    r = fread(&vertexCount, sizeof(unsigned int), 1, file);
    if (r != 1) {
        fprintf(stderr, "Error reading vertex count from file %s\n", filename);
        fclose(file);
        return;
    }
    r = fread(&indexCount, sizeof(unsigned int), 1, file);
    if (r != 1) {
        fprintf(stderr, "Error reading index count from file %s\n", filename);
        fclose(file);
        return;
    }

    // Resize vectors to hold the data
    vertices.resize(vertexCount);
    indices.resize(indexCount);

    // Read vertices
    r = fread(vertices.data(), sizeof(float4), vertexCount, file);
    if (r != vertexCount) {
        fprintf(stderr, "Error reading vertices from file %s\n", filename);
        fclose(file);
        return;
    }

    // Debug: Print first few vertices
    printf("First read vertices:\n");
    for (size_t i = 0; i < std::min(vertexCount, static_cast<unsigned int>(10)); ++i) {
        printf("Vertex %zu: (%f, %f, %f, %f)\n", i, vertices[i].x, vertices[i].y, vertices[i].z, vertices[i].w);
    }

    // Read indices
    r = fread(indices.data(), sizeof(unsigned int), indexCount, file);
    if (r != indexCount) {
        fprintf(stderr, "Error reading indices from file %s\n", filename);
        fclose(file);
        return;
    }

    unsigned char buffer[256];
    while((r = fread(buffer, 1, sizeof(buffer), file)) > 0) {
        fprintf(stderr, "Warning: Extra data found in file %s after reading expected counts:\n", filename);
        fprintf(stderr, "Remaining content (up to 256 bytes):\n");
        for (size_t i = 0; i < r; ++i) {
            fprintf(stderr, "0x%02x ", buffer[i]);
        }
        fprintf(stderr, "\n");
    }

    fclose(file);
}


int main(int argc, char** argv)
{
    // CLI: app [mesh_file]
    const char* meshPath = nullptr;

    if (argc >= 2) {
        if (std::strcmp(argv[1], "-h") == 0 || std::strcmp(argv[1], "--help") == 0) {
            std::cout << "Usage: " << argv[0] << " [mesh_file]\n"
                      << "If mesh_file is provided, the viewer loads triangles from that file.\n";
            return 0;
        }
        meshPath = argv[1];
    } else {
        // Default (optional): keep your old relative path or choose a clearer default
        meshPath = "triangles.bin";
        std::cout << "No mesh file passed. Defaulting to: " << meshPath << "\n";
    }

    GLuint zero = 0;


    static int frameCount = 0;
    static float timeAccum = 0.0f;

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "GPU Triangles", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0); // Disable V-Sync (only for benchmarking)
    gladLoadGL();
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetCursorPosCallback(window, mouse_callback);

    GLuint renderProgram = createProgram("../shader_vert.glsl", "../shader_frag.glsl", "../shader_geom.glsl");

    GLuint verticesSSBO;
    glGenBuffers(1, &verticesSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, verticesSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, MAX_TRIANGLES * sizeof(glm::vec4) * 3, nullptr, GL_DYNAMIC_DRAW); // Approximative upper bound for num of vertices
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, verticesSSBO); // Binding = 2

    GLuint indicesSSBO;
    glGenBuffers(1, &indicesSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, indicesSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, MAX_TRIANGLES * sizeof(glm::uint), nullptr, GL_DYNAMIC_DRAW); // Approximative upper bound for num of indices
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, indicesSSBO); // Binding = 5
    
    GLuint vertexCounterBuffer;
    glGenBuffers(1, &vertexCounterBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, vertexCounterBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint), &zero, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, vertexCounterBuffer); // Binding = 7
    
    GLuint indicesCounterBuffer;
    glGenBuffers(1, &indicesCounterBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, indicesCounterBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint), &zero, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, indicesCounterBuffer); // Binding = 8

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    // Bind vertices
    glBindBuffer(GL_ARRAY_BUFFER, verticesSSBO);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(
        0, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)0
    );

    // Bind indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indicesSSBO);

    GLint projLoc = glGetUniformLocation(renderProgram, "proj");
    GLint viewLoc = glGetUniformLocation(renderProgram, "view");

    float lastFrame = 0.0f;

    while (!glfwWindowShouldClose(window)) {
        float currentFrame = static_cast<float>(glfwGetTime());
        float deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        processInput(window, deltaTime);

        glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BUFFER | GL_SHADER_STORAGE_BARRIER_BIT | GL_ATOMIC_COUNTER_BARRIER_BIT);

        // Read the triangles from file:
        std::vector<float4> vertices;
        std::vector<GLuint> indices;

        readTrianglesFromFile(meshPath, vertices, indices);
        glBindBuffer(GL_ARRAY_BUFFER, verticesSSBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec4), vertices.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indicesSSBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_STATIC_DRAW);

        // Update vertexCount and indicesCount on the GPU
        glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, vertexCounterBuffer);
        unsigned int vertexCountTemp = static_cast<unsigned int>(vertices.size());
        glBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint), &vertexCountTemp);
        glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, indicesCounterBuffer);
        unsigned int indicesCountTemp = static_cast<unsigned int>(indices.size());
        glBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint), &indicesCountTemp);

        GLuint vertexCount = 0, indicesCount = 0;

        glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, vertexCounterBuffer);
        glGetBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint), &vertexCount);

        glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, indicesCounterBuffer);
        glGetBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint), &indicesCount);

        std::cout << "Vertices number: " << vertexCount << std::endl;
        std::cout << "Indices number: " << indicesCount << std::endl;

        // --- Read triangle data back to CPU
        std::vector<glm::vec4> verticesCPU(vertexCount);
        if (vertexCount > 0) {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, verticesSSBO);
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, vertexCount * sizeof(glm::vec4), verticesCPU.data());
        }

        std::vector<glm::uint> indicesCPU(indicesCount);
        if (indicesCount > 0) {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, indicesSSBO);
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, indicesCount * sizeof(glm::uint), indicesCPU.data());
        }

        // Verify that all indices are lower than vertexCount
        if (indicesCount > 0) {
            std::vector<glm::uint> indicesCPU(indicesCount);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, indicesSSBO);
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, indicesCount * sizeof(glm::uint), indicesCPU.data());

            for (const auto& index : indicesCPU) {
                if (index >= vertexCount) {
                    std::cerr << "Error: Index " << index << " is out of bounds (vertexCount = " << vertexCount << ")" << std::endl;
                }
            }
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(renderProgram);

        glm::mat4 proj = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);

        glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(proj));
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));

        glBindVertexArray(vao);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indicesSSBO);
        glDrawElements(
            GL_TRIANGLES,
            indicesCount,
            GL_UNSIGNED_INT,
            (void*)0
        );

        frameCount++;
        timeAccum += deltaTime;

        if (timeAccum >= 1.0f) {
            std::cout << "FPS: " << frameCount << std::endl;
            frameCount = 0;
            timeAccum = 0.0f;
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteBuffers(1, &indicesSSBO);
    glDeleteBuffers(1, &verticesSSBO);
    glDeleteBuffers(1, &indicesCounterBuffer);
    glDeleteBuffers(1, &vertexCounterBuffer);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(renderProgram);
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
