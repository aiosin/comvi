#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "stb_image.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

// Callback functions
void framebufferSizeCallback(GLFWwindow *window, int width, int height);
void scrollCallback(GLFWwindow *window, double offsetX, double offsetY);

void inputProcessing(GLFWwindow *window, float betweenFrames);

// Helper functions
GLint compileShader(GLenum type, const char *path);
GLint linkProgram(std::vector<GLint> shaders);
void detachingDeletingShaders(GLint program, std::vector<GLint> shaders);
void textureLoading(GLuint texture, GLchar* imagePath);
void bufferArrayObjInitialization(GLuint VBO, GLuint VAO, GLuint EBO, 
                                  std::vector<GLfloat> vertices, std::vector<GLint> indices);

static int WINDOW_WIDTH = 1024;
static int WINDOW_HEIGHT = 768;
GLFWwindow *g_window;

// camera
glm::vec3 g_cameraPos = glm::vec3(0.0f, 0.0f, 1.5f);
glm::vec3 g_cameraOppDir = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 g_cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

// For scrollCallback
double g_scrollSensitivity = 0.1;


int main()
{
    glfwInit();

    // Creating GLFW window
    g_window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Cluster Visualization", nullptr, nullptr);
    if (!g_window)
    {
        std::cerr << "Failed to create a window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(g_window);

    
    glfwGetFramebufferSize(g_window, &WINDOW_WIDTH, &WINDOW_HEIGHT);

    // Register callback functions
    glfwSetFramebufferSizeCallback(g_window, framebufferSizeCallback);
    glfwSetScrollCallback(g_window, scrollCallback);

    // Loading OpenGL function pointers
    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
    {
        std::cerr << "Failed to load OpenGL context" << std::endl;
        return -1;
    }

    std::cout << "OpenGL" << glGetString(GL_VERSION) << "\n";
    std::cout << "GLSL" << glGetString(GL_SHADING_LANGUAGE_VERSION) << "\n";

    // Configuration of the global state - if needed
    glEnable(GL_DEPTH_TEST);

    // debug
    std::cout << "before compiling shaders" << std::endl;
    GLint vertexShader = compileShader(GL_VERTEX_SHADER, "shaders/vert.glsl");
    
    GLint fragmentShader = compileShader(GL_FRAGMENT_SHADER, "shaders/frag.glsl");

    // debug
    std::cerr << "after compiling shaders";
    GLint shaderProgram = linkProgram({ vertexShader, fragmentShader });
    detachingDeletingShaders(shaderProgram, { vertexShader, fragmentShader });

    
    std::vector<GLfloat> vertices = {
        0.5f,  0.25f, 0.0f, 1.0f, 1.0f,
        0.5f,  -0.25f, 0.0f, 1.0f, 0.0f,
        -0.5f, -0.25f, 0.0f, 0.0f, 0.0f,
        -0.5f,  0.25f, 0.0f, 0.0f, 1.0f
    };

    std::vector<GLfloat> smallVertices = {
        0.2f,  0.1f, 0.0f, 1.0f, 1.0f,
        0.2f,  -0.1f, 0.0f, 1.0f, 0.0f,
        -0.2f, -0.1f, 0.0f, 0.0f, 0.0f,
        -0.2f,  0.1f, 0.0f, 0.0f, 1.0f
    };

    std::vector<GLint> indices = {
        0, 1, 3,
        1, 2, 3  
    };

    // debug - only temporary - later depending on the clusters
    glm::vec3 clusterPositions[] = {
        glm::vec3(-1.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 1.0f, 0.0f),
        glm::vec3(1.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, -1.0f, 0.0f)
    };

    GLuint VBO, VAO, EBO;
    glGenBuffers(1, &VBO);
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &EBO);

    GLuint sVBO, sVAO, sEBO;
    glGenBuffers(1, &sVBO);
    glGenVertexArrays(1, &sVAO);
    glGenBuffers(1, &sEBO);

    bufferArrayObjInitialization(VBO, VAO, EBO, vertices, indices);
    bufferArrayObjInitialization(sVBO, sVAO, sEBO, smallVertices, indices);

    
    GLuint texture;

    glGenTextures(1, &texture);
    textureLoading(texture, "resources/1EDBA.png");
    
    glUseProgram(shaderProgram);
    glUniform1i(glGetUniformLocation(shaderProgram, "texture"), 0);

    glm::mat4 projection = glm::perspective(glm::pi<float>()/2, 
        static_cast<float>(WINDOW_WIDTH) / static_cast<float>(WINDOW_HEIGHT), 0.1f, 500.0f);    
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "proj"), 1, GL_FALSE, glm::value_ptr(projection));
   
    
    // Timing, to adjust it to the speed of the users' machine
    float betweenFrames = 0.0f;
    float lastFrame = 0.0f;
    
    while (!glfwWindowShouldClose(g_window))
    {
        float currentFrame = glfwGetTime();
        betweenFrames = currentFrame - lastFrame;
        lastFrame = currentFrame;

        inputProcessing(g_window, betweenFrames);

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);
        glUseProgram(shaderProgram);
        

        glm::mat4 view = glm::lookAt(g_cameraPos, g_cameraPos + g_cameraOppDir, g_cameraUp);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));

        if (g_cameraPos.z < 0.5f)
        {
            glBindVertexArray(sVAO);
            for (GLuint i = 0; i < 4; ++i)
            {
                for (GLuint j = 0; j < 4; ++j)
                {
                    glm::mat4 model;
                    model = glm::translate(model, clusterPositions[i] + 0.3f * clusterPositions[j]);
                    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
                    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
                }
            }
        }
        else 
        {
            glBindVertexArray(VAO);
            // debug
            /*for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
            std::cout << "view" << j << " " << view[i][j] << std::endl;
            }
            }*/
            for (GLuint i = 0; i < 4; ++i)
            {
                glm::mat4 model;
                model = glm::translate(model, clusterPositions[i]);
                glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
                glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
            }
        }
        
        glfwSwapBuffers(g_window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteVertexArrays(1, &sVAO);
    glDeleteBuffers(1, &sVBO);
    glDeleteBuffers(1, &sEBO);

    // Terminate GLFW
    glfwTerminate();
    return 0;
}

/**
Reads a file on the given path and returns its content in std::string.

@param path path to a file
@return std::string that contains content of file
*/
std::string readFile(std::string path) 
{
    std::ifstream inf(path);
    char result[MAX_PATH];
    int bytes = GetModuleFileName(NULL, result, MAX_PATH);
    if (!inf)
    {
        std::cerr << path.c_str() << "could not be opened for reading" << std::endl;
    }
    // debug
    std::cout << "Shader source code on path " << path << " read." << std::endl;
    return std::string((std::istreambuf_iterator<char>(inf)),
        (std::istreambuf_iterator<char>()));
}

/**
Checks whether an error occurred during linking. If it occurred, it prints it
to a cerr stream.

@param program reference to a program object
*/
void programLinkingErrorCheck(GLint program)
{
    GLint positiveLinkStatus;
    GLchar infoLog[1024];
    glGetProgramiv(program, GL_LINK_STATUS, &positiveLinkStatus);
    if (!positiveLinkStatus)
    {
        glGetProgramInfoLog(program, 1024, nullptr, infoLog);
        std::cerr << "Failed to link a shader program:\n" << infoLog;
    }
}

/**
Checks whether an error occurred during shader compilation. If it occurred, it
prints it to a cerr stream.

@param shader reference to a shader object
@param type GLenum that specifies the type of the shader
*/
void shaderCompilationErrorCheck(GLint shader, GLenum type)
{
    GLint positiveStatus;
    GLchar infoLog[1024];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &positiveStatus);
    if (!positiveStatus)
    {
        glGetShaderInfoLog(shader, 1024, nullptr, infoLog);
        if (type == GL_VERTEX_SHADER) {
            std::cerr << "Failed to compile a VERTEX SHADER:\n" << infoLog << std::endl;;
        }
        else if (type == GL_FRAGMENT_SHADER) {
            std::cerr << "Failed to compile a FRAGMENT SHADER:\n" << infoLog << std::endl;;
        }
        else {
            std::cerr << "Failed to compile a SHADER:\n" << infoLog << std::endl;
        }
    }
}

/**
Creates a shader object of the given type, compiles it using the source code 
on the given path and returns this shader object.

@param shaders vector of references to shader objects
@return program reference to the program object
*/
GLint compileShader(GLenum type, const char *path)
{
    GLint shader = glCreateShader(type);

    std::string codeString = readFile(path);
    const GLint codeLength = codeString.length();
    const char *shaderCode = codeString.c_str();
    // debug
    // std::cout << shaderCode << std::endl;
    glShaderSource(shader, 1, &shaderCode, &codeLength);
    glCompileShader(shader);
    shaderCompilationErrorCheck(shader, type);
    return shader;
}

/**
Links given shaders into one shader program and returns the corresponding object.

@param shaders vector of references to shader objects
@return program reference to the program object
*/
GLint linkProgram(std::vector<GLint> shaders)
{
    GLint program = glCreateProgram();
    for (GLint shader : shaders)
    {
        glAttachShader(program, shader);
    }
    glLinkProgram(program);
    programLinkingErrorCheck(program);
    return program;
}

// debug - a shader can be linked to a few programs? Maybe shouldn't delete them.
/**
Detaches the given shaders from the program and removes the corresponding shader objects.

@param program texture object id
@param shaders vector of shader objects
*/
void detachingDeletingShaders(GLint program, std::vector<GLint> shaders)
{
    for (GLint shader : shaders)
    {
        glDetachShader(program, shader);
        glDeleteShader(shader);
    }
}

void bufferArrayObjInitialization(GLuint VBO, GLuint VAO, GLuint EBO,
                                  std::vector<GLfloat> vertices, std::vector<GLint> indices)
{
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);

    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * vertices.size(), vertices.data(), GL_STATIC_DRAW);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLint) * indices.size(), indices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
}

/**
Loads texture image on the given image path.

@param texture texture object id
@param imagePath path to image file
*/
void textureLoading(GLuint texture, GLchar *imagePath) {

    glBindTexture(GL_TEXTURE_2D, texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    int texWidth, texHeight, numTexChannels;

    // 0.0 coordinate on the y-axis should in openGL be on the bottom of the image
    stbi_set_flip_vertically_on_load(true);

    GLubyte *image = stbi_load(imagePath, &texWidth, &texHeight, &numTexChannels, 0);
    if (image)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texWidth, texHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, image);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else
    {
        std::cerr << "Texture was not loaded" << std::endl;
    }
    stbi_image_free(image);
}

/**
Callback function that is called when the window size changes.

@param window GLFWwindow object
@param width new width of the window
@param height new height of the window
*/
void framebufferSizeCallback(GLFWwindow *window, int width, int height)
{
    glViewport(0, 0, width, height);
}

/**
Callback function that is called when the user scrolls.
It allows zooming by changing the position of camera on z-axis.

@param window GLFWwindow object
@param width new width of the window
@param height new height of the window
*/
void scrollCallback(GLFWwindow *window, double offsetX, double offsetY)
{
    g_cameraPos.z -= g_scrollSensitivity * offsetY;
    if (g_cameraPos.z < 0.1f) 
    {
        g_cameraPos.z = 0.1f;
    }
    else if (g_cameraPos.z > 500.0f)
    {
        g_cameraPos.z = 500.0f;
    }
}

/**
Processes the input, responsible for the interaction.

@param window GLFWwindow object
@param betweenFrames time between frames
*/
void inputProcessing(GLFWwindow *window, float betweenFrames)
{   
    float cameraSpeed = 3.0f * betweenFrames;

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) 
    {
        g_cameraPos += cameraSpeed * g_cameraUp;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    {
        g_cameraPos -= cameraSpeed * g_cameraUp;
    } 
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    {
        g_cameraPos -= glm::normalize(glm::cross(g_cameraOppDir, g_cameraUp)) * cameraSpeed;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    {
        g_cameraPos += glm::normalize(glm::cross(g_cameraOppDir, g_cameraUp)) * cameraSpeed;
    }

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, true);
    }
}
