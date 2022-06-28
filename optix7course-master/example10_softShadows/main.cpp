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

#include "SampleRenderer.h"

// our helper library for window handling
#include "glfWindow/GLFWindow.h"
#include <GL/gl.h>

#include "LaunchParams.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <string>
#include <cstdlib>
#include <vector>

#include "clip.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define AutoSaveImageAtNthFrame
#ifdef AutoSaveImageAtNthFrame
#define FrameToSave 1151
#endif


/*! \namespace osc - Optix Siggraph Course */
namespace osc
{
    static bool pause = false;
    struct SampleWindow : public GLFCameraWindow
    {
        SampleWindow(const std::string &title,
                     const Model *model,
                     const Camera &camera,
                     const QuadLight &light,
                     const float worldScale)
                : GLFCameraWindow(title, camera.from, camera.at, camera.up, worldScale),
                  sample(model, light)
        {
            sample.setCamera(camera);
        }

        virtual void render() override
        {
            if(!pause)
            {
                if (cameraFrame.modified)
                {
                    sample.setCamera(Camera{cameraFrame.get_from(),
                                            cameraFrame.get_at(),
                                            cameraFrame.get_up()});
                    cameraFrame.modified = false;
                }
                sample.render();
            }
        }

        virtual void imgui() override
        {
            //ImGui::ShowDemoWindow();
            sample.downloadPixels(pixels.data());

            //int w = fbSize.x, h = fbSize.y;
            //PPD *ppd = new PPD[w*h];        //per pixel data
            //sample.downloadPPD(ppd, w, h);  //CONSUME PCIE A LOT

            ImGui::Begin("fps counter");
            ImGui::Text("Frame ID: %d", sample.getAccumID());
            ImGui::Text("Application %.3f ms/frame (%.1f FPS)", ImGui::GetIO().DeltaTime * 1000.f, 1.f / ImGui::GetIO().DeltaTime);
            ImGui::Text("Average of 120Frame %.3f ms/frame (%.1f FPS)", 1000.f/ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::Text("Camera: position: (%f,%f,%f), at(%f,%f,%f)", cameraFrame.get_from().x, cameraFrame.get_from().y, cameraFrame.get_from().z, cameraFrame.get_at().x, cameraFrame.get_at().y, cameraFrame.get_at().z);

            ImGui::SliderInt("spp per frame", &sample.getLaunchParams()->numPixelSamples, 1, 1000);
            if(ImGui::Button("Save Image"))
            {
                std::vector<float> AccuPixels(fbSize.x * 3 * fbSize.y, 0);

                sample.downloadPixelsAccumulate(AccuPixels.data());

                std::string outputPath("../../outputImage/");
                outputPath += std::to_string(sample.getAccumID()) + std::string("th Frame manuallyOutput.hdr");
                std::cout << "outputting file to " << outputPath << std::endl;

                stbi_write_hdr(outputPath.c_str(), fbSize.x, fbSize.y, 3, AccuPixels.data());
            }

            if(ImGui::Button("Copy Image"))
            {
                //handle_copy_image_to_clipboard(pixels.data(), fbSize.x, fbSize.y);

                clip::image_spec spec;
                spec.width = fbSize.x;
                spec.height = fbSize.y;
                spec.bits_per_pixel = 32;
                spec.bytes_per_row = spec.width * 4;
                spec.red_mask = 0x000000ff;
                spec.green_mask = 0x0000ff00;
                spec.blue_mask = 0x00ff0000;
                spec.alpha_mask = 0xff000000;
                spec.red_shift = 0;
                spec.green_shift = 8;
                spec.blue_shift = 16;
                spec.alpha_shift = 24;
                const clip::image img(pixels.data(), spec);
                clip::set_image(img);
            }
            if(ImGui::Button("Pause / play"))
            {
                pause = !pause;
            }
            ImGui::End();

#ifdef AutoSaveImageAtNthFrame
            if(sample.getAccumID() == FrameToSave)
            {
                std::vector<float> AccuPixels(fbSize.x * 3 * fbSize.y, 0);

                sample.downloadPixelsAccumulate(AccuPixels.data());

                std::string outputPath("../../outputImage/");
                outputPath += std::to_string(sample.getAccumID()) + std::string("th Frame output.hdr");
                std::cout << "outputting file to " << outputPath << std::endl;

                stbi_write_hdr(outputPath.c_str(), fbSize.x, fbSize.y, 3, AccuPixels.data());
            }
#endif
            //delete[] ppd;
        }

        virtual void draw() override
        {

            if (fbTexture == 0)
                glGenTextures(1, &fbTexture);

            glBindTexture(GL_TEXTURE_2D, fbTexture);
            GLenum texFormat = GL_RGBA;
            GLenum texelType = GL_UNSIGNED_BYTE;
            glTexImage2D(GL_TEXTURE_2D, 0, texFormat, fbSize.x, fbSize.y, 0, GL_RGBA,
                         texelType, pixels.data());

            glDisable(GL_LIGHTING);
            glColor3f(1, 1, 1);

            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();

            glEnable(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, fbTexture);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            glDisable(GL_DEPTH_TEST);

            glViewport(0, 0, fbSize.x, fbSize.y);

            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            glOrtho(0.f, (float) fbSize.x, (float) fbSize.y, 0.f, -1.f, 1.f);

            glBegin(GL_QUADS);
            {
                glTexCoord2f(0.f, 0.f);
                glVertex3f(0.f, 0.f, 0.f);

                glTexCoord2f(0.f, 1.f);
                glVertex3f(0.f, (float) fbSize.y, 0.f);

                glTexCoord2f(1.f, 1.f);
                glVertex3f((float) fbSize.x, (float) fbSize.y, 0.f);

                glTexCoord2f(1.f, 0.f);
                glVertex3f((float) fbSize.x, 0.f, 0.f);
            }
            glEnd();
        }

        virtual void resize(const vec2i &newSize)
        {
            fbSize = newSize;
            sample.resize(newSize);
            pixels.resize(newSize.x * newSize.y);
        }

        vec2i fbSize;
        GLuint fbTexture{0};
        SampleRenderer sample;
        std::vector<uint32_t> pixels;
    };


    /*! main entry point to this example - initially optix, print hello
      world, then exit */
    extern "C" int main(int ac, char **av)
    {
        try
        {
            Model *model = loadOBJ(
#ifdef _WIN32
                    // on windows, visual studio creates _two_ levels of build dir
                    // (x86/Release)
                   // "../../models/sponza/sponza.obj"
                    "../../models/bathroom/salle_de_bain.obj"
                    //"../../models/eggtest/eggtest.obj"
                    //"../../models/cornellBoxTeapot/teapotbox.obj"
                    //"../../models/benchmarkModels/Scene_high_tri_2_3.obj" //dragonBall
                    //"../../models/Scene_high_obj_2_2.obj" //waterDrop
                    //"../../models/torus.obj" //torus
                    //"../../models/one_light_DSL.obj" //waterDrop
                    //"../../models/prism/80.obj" //prism
                    //"../../models/eggSpectral/egg_experiment.obj"
                    //"../../models/boundtest.obj" //prism
#else
                    // on linux, common practice is to have ONE level of build dir
                    // (say, <project>/build/)...
                    "../models/sponza.obj"
#endif
            );

            Camera camera = {vec3f(0.0f, 6.0f, 25.0f), vec3f(0.0f, 3.0f, 0.0f), vec3f(0.0f, 1.0f, 0.0f)}; //dragonBall
            //Camera camera = {vec3f(0.0f, 3.5f, 50.0f), vec3f(0.0f, 5.0f, 0.0f), vec3f(0.0f, 1.0f, 0.0f)}; //waterDrop
            //Camera camera = {vec3f(0.f, 1.f, 30.0f), vec3f(0.0f, 0.0f, 0.0f), vec3f(0.0f, 1.0f, 0.0f)}; //torus
            //Camera camera = {vec3f(59.58f, 13.55f, 4.25f), vec3f(11.22f, 5.08f, -5.366f), vec3f(0.0f, 1.0f, 0.0f)}; //prism
            //Camera camera = {vec3f(30.f, 20.f, -60.f), vec3f(11.22f, 5.08f, -5.366f), vec3f(0.0f, 1.0f, 0.0f)}; //egg_experiment.obj
            //Camera camera = { /*from*/vec3f(0.f, 17.f, 70.f),
            //        /* at */ vec3f(0.f, 17.0f, 65), //model->bounds.center()
            //        /* up */vec3f(0.f, 1.f, 0.f)};

            //Camera camera = { /*from*/vec3f(16.f, 4.f, 0.f),
            //        /* at */ vec3f(0.f, 4.0f, 0.f), //model->bounds.center()
            //        /* up */vec3f(0.f, 1.f, 0.f)};

            // some simple, hard-coded light ... obviously, only works for sponza
            const float light_size = 100.f;
            QuadLight light = { /* origin */ vec3f(-1300 - light_size, 50, light_size),
                    /* edge 1 */ vec3f(2.f * light_size, 0, 0),
                    /* edge 2 */ vec3f(0, 0, 2.f * light_size),
                    /* power */  vec3f(300000.f)};

            // something approximating the scale of the world, so the
            // camera knows how much to move for any given user interaction:
            const float worldScale = length(model->bounds.span());

            SampleWindow *window = new SampleWindow("Optix 7 Course Example",
                                                    model, camera, light, worldScale);
            window->enableFlyMode();
            //window->enableInspectMode();
            window->run();

        } catch (std::runtime_error &e)
        {
            std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what()
                      << GDT_TERMINAL_DEFAULT << std::endl;
            std::cout << "Did you forget to copy sponza.obj and sponza.mtl into your optix7course/models directory?" << std::endl;
            exit(1);
        }
        return 0;
    }

} // ::osc
