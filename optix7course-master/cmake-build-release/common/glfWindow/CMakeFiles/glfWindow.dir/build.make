# CMAKE generated file: DO NOT EDIT!
# Generated by "NMake Makefiles" Generator, CMake Version 3.19

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE
NULL=nul
!ENDIF
SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "E:\appdata\CLION\CLion 2021.1.2\bin\cmake\win\bin\cmake.exe"

# The command to remove a file.
RM = "E:\appdata\CLION\CLion 2021.1.2\bin\cmake\win\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = D:\Codes\Optix\optix7PT_codeBase\optix7course-master

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release

# Include any dependencies generated for this target.
include common\glfWindow\CMakeFiles\glfWindow.dir\depend.make

# Include the progress variables for this target.
include common\glfWindow\CMakeFiles\glfWindow.dir\progress.make

# Include the compile flags for this target's objects.
include common\glfWindow\CMakeFiles\glfWindow.dir\flags.make

common\glfWindow\CMakeFiles\glfWindow.dir\GLFWindow.cpp.obj: common\glfWindow\CMakeFiles\glfWindow.dir\flags.make
common\glfWindow\CMakeFiles\glfWindow.dir\GLFWindow.cpp.obj: ..\common\glfWindow\GLFWindow.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object common/glfWindow/CMakeFiles/glfWindow.dir/GLFWindow.cpp.obj"
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release\common\glfWindow
	E:\appdata\VS2019IDE\VC\Tools\MSVC\14.29.30037\bin\Hostx64\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoCMakeFiles\glfWindow.dir\GLFWindow.cpp.obj /FdCMakeFiles\glfWindow.dir\glfWindow.pdb /FS -c D:\Codes\Optix\optix7PT_codeBase\optix7course-master\common\glfWindow\GLFWindow.cpp
<<
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release

common\glfWindow\CMakeFiles\glfWindow.dir\GLFWindow.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/glfWindow.dir/GLFWindow.cpp.i"
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release\common\glfWindow
	E:\appdata\VS2019IDE\VC\Tools\MSVC\14.29.30037\bin\Hostx64\x64\cl.exe > CMakeFiles\glfWindow.dir\GLFWindow.cpp.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\Codes\Optix\optix7PT_codeBase\optix7course-master\common\glfWindow\GLFWindow.cpp
<<
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release

common\glfWindow\CMakeFiles\glfWindow.dir\GLFWindow.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/glfWindow.dir/GLFWindow.cpp.s"
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release\common\glfWindow
	E:\appdata\VS2019IDE\VC\Tools\MSVC\14.29.30037\bin\Hostx64\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\glfWindow.dir\GLFWindow.cpp.s /c D:\Codes\Optix\optix7PT_codeBase\optix7course-master\common\glfWindow\GLFWindow.cpp
<<
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release

common\glfWindow\CMakeFiles\glfWindow.dir\__\imgui\backends\imgui_impl_glfw.cpp.obj: common\glfWindow\CMakeFiles\glfWindow.dir\flags.make
common\glfWindow\CMakeFiles\glfWindow.dir\__\imgui\backends\imgui_impl_glfw.cpp.obj: ..\common\imgui\backends\imgui_impl_glfw.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object common/glfWindow/CMakeFiles/glfWindow.dir/__/imgui/backends/imgui_impl_glfw.cpp.obj"
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release\common\glfWindow
	E:\appdata\VS2019IDE\VC\Tools\MSVC\14.29.30037\bin\Hostx64\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoCMakeFiles\glfWindow.dir\__\imgui\backends\imgui_impl_glfw.cpp.obj /FdCMakeFiles\glfWindow.dir\glfWindow.pdb /FS -c D:\Codes\Optix\optix7PT_codeBase\optix7course-master\common\imgui\backends\imgui_impl_glfw.cpp
<<
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release

common\glfWindow\CMakeFiles\glfWindow.dir\__\imgui\backends\imgui_impl_glfw.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/glfWindow.dir/__/imgui/backends/imgui_impl_glfw.cpp.i"
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release\common\glfWindow
	E:\appdata\VS2019IDE\VC\Tools\MSVC\14.29.30037\bin\Hostx64\x64\cl.exe > CMakeFiles\glfWindow.dir\__\imgui\backends\imgui_impl_glfw.cpp.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\Codes\Optix\optix7PT_codeBase\optix7course-master\common\imgui\backends\imgui_impl_glfw.cpp
<<
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release

common\glfWindow\CMakeFiles\glfWindow.dir\__\imgui\backends\imgui_impl_glfw.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/glfWindow.dir/__/imgui/backends/imgui_impl_glfw.cpp.s"
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release\common\glfWindow
	E:\appdata\VS2019IDE\VC\Tools\MSVC\14.29.30037\bin\Hostx64\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\glfWindow.dir\__\imgui\backends\imgui_impl_glfw.cpp.s /c D:\Codes\Optix\optix7PT_codeBase\optix7course-master\common\imgui\backends\imgui_impl_glfw.cpp
<<
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release

common\glfWindow\CMakeFiles\glfWindow.dir\__\imgui\backends\imgui_impl_opengl3.cpp.obj: common\glfWindow\CMakeFiles\glfWindow.dir\flags.make
common\glfWindow\CMakeFiles\glfWindow.dir\__\imgui\backends\imgui_impl_opengl3.cpp.obj: ..\common\imgui\backends\imgui_impl_opengl3.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release\CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object common/glfWindow/CMakeFiles/glfWindow.dir/__/imgui/backends/imgui_impl_opengl3.cpp.obj"
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release\common\glfWindow
	E:\appdata\VS2019IDE\VC\Tools\MSVC\14.29.30037\bin\Hostx64\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoCMakeFiles\glfWindow.dir\__\imgui\backends\imgui_impl_opengl3.cpp.obj /FdCMakeFiles\glfWindow.dir\glfWindow.pdb /FS -c D:\Codes\Optix\optix7PT_codeBase\optix7course-master\common\imgui\backends\imgui_impl_opengl3.cpp
<<
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release

common\glfWindow\CMakeFiles\glfWindow.dir\__\imgui\backends\imgui_impl_opengl3.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/glfWindow.dir/__/imgui/backends/imgui_impl_opengl3.cpp.i"
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release\common\glfWindow
	E:\appdata\VS2019IDE\VC\Tools\MSVC\14.29.30037\bin\Hostx64\x64\cl.exe > CMakeFiles\glfWindow.dir\__\imgui\backends\imgui_impl_opengl3.cpp.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\Codes\Optix\optix7PT_codeBase\optix7course-master\common\imgui\backends\imgui_impl_opengl3.cpp
<<
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release

common\glfWindow\CMakeFiles\glfWindow.dir\__\imgui\backends\imgui_impl_opengl3.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/glfWindow.dir/__/imgui/backends/imgui_impl_opengl3.cpp.s"
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release\common\glfWindow
	E:\appdata\VS2019IDE\VC\Tools\MSVC\14.29.30037\bin\Hostx64\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\glfWindow.dir\__\imgui\backends\imgui_impl_opengl3.cpp.s /c D:\Codes\Optix\optix7PT_codeBase\optix7course-master\common\imgui\backends\imgui_impl_opengl3.cpp
<<
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release

common\glfWindow\CMakeFiles\glfWindow.dir\__\imgui\imgui.cpp.obj: common\glfWindow\CMakeFiles\glfWindow.dir\flags.make
common\glfWindow\CMakeFiles\glfWindow.dir\__\imgui\imgui.cpp.obj: ..\common\imgui\imgui.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release\CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object common/glfWindow/CMakeFiles/glfWindow.dir/__/imgui/imgui.cpp.obj"
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release\common\glfWindow
	E:\appdata\VS2019IDE\VC\Tools\MSVC\14.29.30037\bin\Hostx64\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoCMakeFiles\glfWindow.dir\__\imgui\imgui.cpp.obj /FdCMakeFiles\glfWindow.dir\glfWindow.pdb /FS -c D:\Codes\Optix\optix7PT_codeBase\optix7course-master\common\imgui\imgui.cpp
<<
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release

common\glfWindow\CMakeFiles\glfWindow.dir\__\imgui\imgui.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/glfWindow.dir/__/imgui/imgui.cpp.i"
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release\common\glfWindow
	E:\appdata\VS2019IDE\VC\Tools\MSVC\14.29.30037\bin\Hostx64\x64\cl.exe > CMakeFiles\glfWindow.dir\__\imgui\imgui.cpp.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\Codes\Optix\optix7PT_codeBase\optix7course-master\common\imgui\imgui.cpp
<<
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release

common\glfWindow\CMakeFiles\glfWindow.dir\__\imgui\imgui.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/glfWindow.dir/__/imgui/imgui.cpp.s"
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release\common\glfWindow
	E:\appdata\VS2019IDE\VC\Tools\MSVC\14.29.30037\bin\Hostx64\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\glfWindow.dir\__\imgui\imgui.cpp.s /c D:\Codes\Optix\optix7PT_codeBase\optix7course-master\common\imgui\imgui.cpp
<<
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release

common\glfWindow\CMakeFiles\glfWindow.dir\__\imgui\imgui_draw.cpp.obj: common\glfWindow\CMakeFiles\glfWindow.dir\flags.make
common\glfWindow\CMakeFiles\glfWindow.dir\__\imgui\imgui_draw.cpp.obj: ..\common\imgui\imgui_draw.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release\CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object common/glfWindow/CMakeFiles/glfWindow.dir/__/imgui/imgui_draw.cpp.obj"
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release\common\glfWindow
	E:\appdata\VS2019IDE\VC\Tools\MSVC\14.29.30037\bin\Hostx64\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoCMakeFiles\glfWindow.dir\__\imgui\imgui_draw.cpp.obj /FdCMakeFiles\glfWindow.dir\glfWindow.pdb /FS -c D:\Codes\Optix\optix7PT_codeBase\optix7course-master\common\imgui\imgui_draw.cpp
<<
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release

common\glfWindow\CMakeFiles\glfWindow.dir\__\imgui\imgui_draw.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/glfWindow.dir/__/imgui/imgui_draw.cpp.i"
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release\common\glfWindow
	E:\appdata\VS2019IDE\VC\Tools\MSVC\14.29.30037\bin\Hostx64\x64\cl.exe > CMakeFiles\glfWindow.dir\__\imgui\imgui_draw.cpp.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\Codes\Optix\optix7PT_codeBase\optix7course-master\common\imgui\imgui_draw.cpp
<<
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release

common\glfWindow\CMakeFiles\glfWindow.dir\__\imgui\imgui_draw.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/glfWindow.dir/__/imgui/imgui_draw.cpp.s"
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release\common\glfWindow
	E:\appdata\VS2019IDE\VC\Tools\MSVC\14.29.30037\bin\Hostx64\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\glfWindow.dir\__\imgui\imgui_draw.cpp.s /c D:\Codes\Optix\optix7PT_codeBase\optix7course-master\common\imgui\imgui_draw.cpp
<<
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release

common\glfWindow\CMakeFiles\glfWindow.dir\__\imgui\imgui_demo.cpp.obj: common\glfWindow\CMakeFiles\glfWindow.dir\flags.make
common\glfWindow\CMakeFiles\glfWindow.dir\__\imgui\imgui_demo.cpp.obj: ..\common\imgui\imgui_demo.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release\CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object common/glfWindow/CMakeFiles/glfWindow.dir/__/imgui/imgui_demo.cpp.obj"
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release\common\glfWindow
	E:\appdata\VS2019IDE\VC\Tools\MSVC\14.29.30037\bin\Hostx64\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoCMakeFiles\glfWindow.dir\__\imgui\imgui_demo.cpp.obj /FdCMakeFiles\glfWindow.dir\glfWindow.pdb /FS -c D:\Codes\Optix\optix7PT_codeBase\optix7course-master\common\imgui\imgui_demo.cpp
<<
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release

common\glfWindow\CMakeFiles\glfWindow.dir\__\imgui\imgui_demo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/glfWindow.dir/__/imgui/imgui_demo.cpp.i"
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release\common\glfWindow
	E:\appdata\VS2019IDE\VC\Tools\MSVC\14.29.30037\bin\Hostx64\x64\cl.exe > CMakeFiles\glfWindow.dir\__\imgui\imgui_demo.cpp.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\Codes\Optix\optix7PT_codeBase\optix7course-master\common\imgui\imgui_demo.cpp
<<
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release

common\glfWindow\CMakeFiles\glfWindow.dir\__\imgui\imgui_demo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/glfWindow.dir/__/imgui/imgui_demo.cpp.s"
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release\common\glfWindow
	E:\appdata\VS2019IDE\VC\Tools\MSVC\14.29.30037\bin\Hostx64\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\glfWindow.dir\__\imgui\imgui_demo.cpp.s /c D:\Codes\Optix\optix7PT_codeBase\optix7course-master\common\imgui\imgui_demo.cpp
<<
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release

common\glfWindow\CMakeFiles\glfWindow.dir\__\imgui\imgui_tables.cpp.obj: common\glfWindow\CMakeFiles\glfWindow.dir\flags.make
common\glfWindow\CMakeFiles\glfWindow.dir\__\imgui\imgui_tables.cpp.obj: ..\common\imgui\imgui_tables.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release\CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object common/glfWindow/CMakeFiles/glfWindow.dir/__/imgui/imgui_tables.cpp.obj"
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release\common\glfWindow
	E:\appdata\VS2019IDE\VC\Tools\MSVC\14.29.30037\bin\Hostx64\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoCMakeFiles\glfWindow.dir\__\imgui\imgui_tables.cpp.obj /FdCMakeFiles\glfWindow.dir\glfWindow.pdb /FS -c D:\Codes\Optix\optix7PT_codeBase\optix7course-master\common\imgui\imgui_tables.cpp
<<
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release

common\glfWindow\CMakeFiles\glfWindow.dir\__\imgui\imgui_tables.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/glfWindow.dir/__/imgui/imgui_tables.cpp.i"
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release\common\glfWindow
	E:\appdata\VS2019IDE\VC\Tools\MSVC\14.29.30037\bin\Hostx64\x64\cl.exe > CMakeFiles\glfWindow.dir\__\imgui\imgui_tables.cpp.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\Codes\Optix\optix7PT_codeBase\optix7course-master\common\imgui\imgui_tables.cpp
<<
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release

common\glfWindow\CMakeFiles\glfWindow.dir\__\imgui\imgui_tables.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/glfWindow.dir/__/imgui/imgui_tables.cpp.s"
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release\common\glfWindow
	E:\appdata\VS2019IDE\VC\Tools\MSVC\14.29.30037\bin\Hostx64\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\glfWindow.dir\__\imgui\imgui_tables.cpp.s /c D:\Codes\Optix\optix7PT_codeBase\optix7course-master\common\imgui\imgui_tables.cpp
<<
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release

common\glfWindow\CMakeFiles\glfWindow.dir\__\imgui\imgui_widgets.cpp.obj: common\glfWindow\CMakeFiles\glfWindow.dir\flags.make
common\glfWindow\CMakeFiles\glfWindow.dir\__\imgui\imgui_widgets.cpp.obj: ..\common\imgui\imgui_widgets.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release\CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object common/glfWindow/CMakeFiles/glfWindow.dir/__/imgui/imgui_widgets.cpp.obj"
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release\common\glfWindow
	E:\appdata\VS2019IDE\VC\Tools\MSVC\14.29.30037\bin\Hostx64\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoCMakeFiles\glfWindow.dir\__\imgui\imgui_widgets.cpp.obj /FdCMakeFiles\glfWindow.dir\glfWindow.pdb /FS -c D:\Codes\Optix\optix7PT_codeBase\optix7course-master\common\imgui\imgui_widgets.cpp
<<
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release

common\glfWindow\CMakeFiles\glfWindow.dir\__\imgui\imgui_widgets.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/glfWindow.dir/__/imgui/imgui_widgets.cpp.i"
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release\common\glfWindow
	E:\appdata\VS2019IDE\VC\Tools\MSVC\14.29.30037\bin\Hostx64\x64\cl.exe > CMakeFiles\glfWindow.dir\__\imgui\imgui_widgets.cpp.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\Codes\Optix\optix7PT_codeBase\optix7course-master\common\imgui\imgui_widgets.cpp
<<
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release

common\glfWindow\CMakeFiles\glfWindow.dir\__\imgui\imgui_widgets.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/glfWindow.dir/__/imgui/imgui_widgets.cpp.s"
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release\common\glfWindow
	E:\appdata\VS2019IDE\VC\Tools\MSVC\14.29.30037\bin\Hostx64\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\glfWindow.dir\__\imgui\imgui_widgets.cpp.s /c D:\Codes\Optix\optix7PT_codeBase\optix7course-master\common\imgui\imgui_widgets.cpp
<<
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release

# Object files for target glfWindow
glfWindow_OBJECTS = \
"CMakeFiles\glfWindow.dir\GLFWindow.cpp.obj" \
"CMakeFiles\glfWindow.dir\__\imgui\backends\imgui_impl_glfw.cpp.obj" \
"CMakeFiles\glfWindow.dir\__\imgui\backends\imgui_impl_opengl3.cpp.obj" \
"CMakeFiles\glfWindow.dir\__\imgui\imgui.cpp.obj" \
"CMakeFiles\glfWindow.dir\__\imgui\imgui_draw.cpp.obj" \
"CMakeFiles\glfWindow.dir\__\imgui\imgui_demo.cpp.obj" \
"CMakeFiles\glfWindow.dir\__\imgui\imgui_tables.cpp.obj" \
"CMakeFiles\glfWindow.dir\__\imgui\imgui_widgets.cpp.obj"

# External object files for target glfWindow
glfWindow_EXTERNAL_OBJECTS =

glfWindow.lib: common\glfWindow\CMakeFiles\glfWindow.dir\GLFWindow.cpp.obj
glfWindow.lib: common\glfWindow\CMakeFiles\glfWindow.dir\__\imgui\backends\imgui_impl_glfw.cpp.obj
glfWindow.lib: common\glfWindow\CMakeFiles\glfWindow.dir\__\imgui\backends\imgui_impl_opengl3.cpp.obj
glfWindow.lib: common\glfWindow\CMakeFiles\glfWindow.dir\__\imgui\imgui.cpp.obj
glfWindow.lib: common\glfWindow\CMakeFiles\glfWindow.dir\__\imgui\imgui_draw.cpp.obj
glfWindow.lib: common\glfWindow\CMakeFiles\glfWindow.dir\__\imgui\imgui_demo.cpp.obj
glfWindow.lib: common\glfWindow\CMakeFiles\glfWindow.dir\__\imgui\imgui_tables.cpp.obj
glfWindow.lib: common\glfWindow\CMakeFiles\glfWindow.dir\__\imgui\imgui_widgets.cpp.obj
glfWindow.lib: common\glfWindow\CMakeFiles\glfWindow.dir\build.make
glfWindow.lib: common\glfWindow\CMakeFiles\glfWindow.dir\objects1.rsp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release\CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Linking CXX static library ..\..\glfWindow.lib"
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release\common\glfWindow
	$(CMAKE_COMMAND) -P CMakeFiles\glfWindow.dir\cmake_clean_target.cmake
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release\common\glfWindow
	E:\appdata\VS2019IDE\VC\Tools\MSVC\14.29.30037\bin\Hostx64\x64\lib.exe /nologo /machine:x64 /out:..\..\glfWindow.lib @CMakeFiles\glfWindow.dir\objects1.rsp 
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release

# Rule to build all files generated by this target.
common\glfWindow\CMakeFiles\glfWindow.dir\build: glfWindow.lib

.PHONY : common\glfWindow\CMakeFiles\glfWindow.dir\build

common\glfWindow\CMakeFiles\glfWindow.dir\clean:
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release\common\glfWindow
	$(CMAKE_COMMAND) -P CMakeFiles\glfWindow.dir\cmake_clean.cmake
	cd D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release
.PHONY : common\glfWindow\CMakeFiles\glfWindow.dir\clean

common\glfWindow\CMakeFiles\glfWindow.dir\depend:
	$(CMAKE_COMMAND) -E cmake_depends "NMake Makefiles" D:\Codes\Optix\optix7PT_codeBase\optix7course-master D:\Codes\Optix\optix7PT_codeBase\optix7course-master\common\glfWindow D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release\common\glfWindow D:\Codes\Optix\optix7PT_codeBase\optix7course-master\cmake-build-release\common\glfWindow\CMakeFiles\glfWindow.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : common\glfWindow\CMakeFiles\glfWindow.dir\depend
