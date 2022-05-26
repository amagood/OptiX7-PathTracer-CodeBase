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
#include <GL/glew.h>
#include "GLFWindow.h"




/*! \namespace osc - Optix Siggraph Course */
namespace osc
{
    using namespace gdt;

    static void glfw_error_callback(int error, const char *description)
    {
        fprintf(stderr, "Error: %s\n", description);
    }

    GLFWindow::~GLFWindow()
    {
        glfwDestroyWindow(handle);
        glfwTerminate();
    }

    GLFWindow::GLFWindow(const std::string &title)
    {
        glfwSetErrorCallback(glfw_error_callback);
        // glfwInitHint(GLFW_COCOA_MENUBAR, GLFW_FALSE);

        if (!glfwInit())
            exit(EXIT_FAILURE);

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
        glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
        glfwWindowHint(GLFW_FOCUSED, GLFW_TRUE);

        handle = glfwCreateWindow(1200, 800, title.c_str(), NULL, NULL);
        if (!handle)
        {
            glfwTerminate();
            exit(EXIT_FAILURE);
        }

        glfwSetWindowUserPointer(handle, this);
        glfwMakeContextCurrent(handle);
        glfwSwapInterval(0);
    }

    /*! callback for a window resizing event */
    static void glfwindow_reshape_cb(GLFWwindow *window, int width, int height)
    {
        GLFWindow *gw = static_cast<GLFWindow *>(glfwGetWindowUserPointer(window));
        assert(gw);
        gw->resize(vec2i(width, height));
        // assert(GLFWindow::current);
        //   GLFWindow::current->resize(vec2i(width,height));
    }

    /*! callback for a key press */
    static void glfwindow_key_cb(GLFWwindow *window, int key, int scancode, int action, int mods)
    {
        GLFWindow *gw = static_cast<GLFWindow *>(glfwGetWindowUserPointer(window));
        assert(gw);
        if (action == GLFW_PRESS)
        {
            gw->key(key, mods);
        }
    }

    /*! callback for _moving_ the mouse to a new position */
    static void glfwindow_mouseMotion_cb(GLFWwindow *window, double x, double y)
    {
        GLFWindow *gw = static_cast<GLFWindow *>(glfwGetWindowUserPointer(window));
        assert(gw);
        gw->mouseMotion(vec2i((int) x, (int) y));
    }

    /*! callback for pressing _or_ releasing a mouse button*/
    static void glfwindow_mouseButton_cb(GLFWwindow *window, int button, int action, int mods)
    {
        GLFWindow *gw = static_cast<GLFWindow *>(glfwGetWindowUserPointer(window));
        assert(gw);
        // double x, y;
        // glfwGetCursorPos(window,&x,&y);
        gw->mouseButton(button, action, mods);
    }

    void GLFWindow::run()
    {
        // Initialize GLEW
        if (glewInit() != GLEW_OK)
        {
            fprintf(stderr, "Failed to initialize GLEW\n");
            getchar();
            glfwTerminate();
            return;
        }

        int width, height;
        glfwGetFramebufferSize(handle, &width, &height);
        resize(vec2i(width, height));

        // glfwSetWindowUserPointer(window, GLFWindow::current);
        glfwSetFramebufferSizeCallback(handle, glfwindow_reshape_cb);
        glfwSetMouseButtonCallback(handle, glfwindow_mouseButton_cb);
        glfwSetKeyCallback(handle, glfwindow_key_cb);
        glfwSetCursorPosCallback(handle, glfwindow_mouseMotion_cb);
        glfwWindowHint( GLFW_REFRESH_RATE, GLFW_DONT_CARE);
        glfwSwapInterval(0);

        //imgui
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO &io = ImGui::GetIO();
        (void) io;
        //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        //io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
        io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;


        ImGui::StyleColorsDark();
        ImGui_ImplGlfw_InitForOpenGL(handle, true);
        ImGui_ImplOpenGL3_Init("#version 130");


        while (!glfwWindowShouldClose(handle))
        {
            render();

            // Start the Dear ImGui frame
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            imgui();
            draw();
            // Rendering
            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());



            // Update and Render additional Platform Windows
            if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
            {
                GLFWwindow* backup_current_context = glfwGetCurrentContext();
                ImGui::UpdatePlatformWindows();
                ImGui::RenderPlatformWindowsDefault();
                glfwMakeContextCurrent(backup_current_context);
            }

            glfwSwapBuffers(handle);
            glfwPollEvents();
        }
    }

    // GLFWindow *GLFWindow::current = nullptr;

} // ::osc