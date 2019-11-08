// In a Catch project with multiple files, dedicate one file to compile the
// source code of Catch itself and reuse the resulting object file for linking.
// Should be defined only once.
#define CATCH_CONFIG_MAIN

#include "../extern/catch/catch.h"

// This file should include NO tests.
