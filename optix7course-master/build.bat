@echo off
setlocal enabledelayedexpansion

:: config

:: buildDir       : the CMake working directory
:: buildType      : the CMake build type                                (Release / Debug / ...)
:: cmakeGenerator : the CMake generator                                 ("MinGW Makefiles" / "NMake Makefiles" / ...)
:: initMsvcEnv    : whether to init MSVC environment                    (1 / 0)
:: parallelBuild  : whether to build parallelly (not works on NMake)    (1 / 0)

set buildDir=build
set buildType=Release
set cmakeGenerator="NMake Makefiles"
set initMsvcEnv=1
set parallelBuild=1

:: compute other variables

set cmakeGenerateCommand=cmake -G !cmakeGenerator! -DCMAKE_BUILD_TYPE=!buildType! ..
set cmakeBuildCommand=cmake --build .
if !parallelBuild! equ 1 (
    set cmakeBuildCommand=!cmakeBuildCommand! -j
)

:: init MSVC environment

if !initMsvcEnv! equ 1 (
    for /f "tokens=*" %%i in ('where cl.exe') do set vcvarsBat=%%i
    set vcvarsBat="!vcvarsBat!\..\..\..\..\..\..\..\Auxiliary\Build\vcvars64.bat"
    call !vcvarsBat!
)

:: make build directory

if not exist !buildDir! (
    echo Making build directory...
    mkdir !buildDir!
)
pushd !buildDir!

:: dectet whether the path has changed

if not exist CMakeCache.txt goto :noClearCache
for /f "tokens=*" %%i in ('findstr /L "CMAKE_CACHEFILE_DIR:INTERNAL=" CMakeCache.txt') do set cmakeCachefileDir=%%i
set "cmakeCachefileDir=!cmakeCachefileDir:~29!"
set "cmakeCachefileDir=!cmakeCachefileDir:/=\!"
if /I not !cmakeCachefileDir!==!cd! (
    echo Build directory path has been changed. Removing old cache...
    del /F CMakeCache.txt
)
:noClearCache

:: generate CMake build system

if not exist CMakeCache.txt (
    echo Generating CMake Files...
    !cmakeGenerateCommand!
)

:: build

echo Building...
!cmakeBuildCommand!

:: exit

popd
exit /b !errorlevel!
