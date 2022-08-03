# ECE-7410 Final Project: Playing Card Generator 

# Build Notes
Note: On windows, ensure OpenCV_ROOT points to a directory containing a 
`OpenCVConfig.cmake` file. 

Also, the Visual Studio CMake generator has not been setting the include directories
for code navigation and autofill correctly, this can be set manually to OpenCV_ROOT/include.

```
mkdir build
cd build
cmake ..
```