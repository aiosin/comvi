#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "stb_image.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>

// Callback functions
// void framebufferSizeCallback(GLFWwindow *window, int width, int height);

// void inputProcessing(GLFWwindow *window);

const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;
GLFWwindow* g_window;

int notMain()
{
	// Initialization of GLFW
	glfwInit();
	

	// Creating GLFW window
	g_window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Cluster Visualization", NULL, NULL);
	if (!g_window)
	{
		std::cout << "Failed to create a window" << std::endl;
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(g_window);
	// glfwSetFramebufferSizeCallback(g_window, framebufferSizeCallback);

	// Loading OpenGL function pointerss
	if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
	{
		std::cout << "Failed to load OpenGL context" << std::endl;
		return -1;
	}

	std::cout << "OpenGL" << glGetString(GL_VERSION) << "\n";
	std::cout << "GLSL" << glGetString(GL_SHADING_LANGUAGE_VERSION) << "\n";

	// Configuration of the global state
	glEnable(GL_DEPTH_TEST);
	

	return 0;
}