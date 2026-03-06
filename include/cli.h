#pragma once
#include <string>
#include <iostream>
#include <cstdlib>

struct Options {
    std::string input  = "";
    float iso = 0.0f;

    int Nx = 128, Ny = 128, Nz = 128;

    // field options
    bool planeMode = false;
    float planeY = 0.0f;
    unsigned int seed = 1;

    // benchmarking
    int warmup = 1;
    int iters  = 10;

    // toggles
    bool useChunks = false;
    bool cullChunks = false;
    bool tiled = false;

    // CLI flags
    bool deterministic = false;
    bool gpuField = false;

    std::string output = "triangles.txt";
};


static inline void printHelp(const char* prog) {
    std::cout <<
    "Usage: ./generator [options]\n"
    "\n"
    "Core:\n"
    "  --output FILE        Output triangles file (default: triangles.txt)\n"
    "  --iso V              Iso value (default: 0.0)\n"
    "  --warmup N           Warmup iterations (default: 5)\n"
    "  --iters N            Timed iterations (default: 50)\n"
    "  --deterministic      Fixed seed for reproducible runs\n"
    "\n"
    "Optimizations:\n"
    "  --cull               Enable chunk culling (OPT2)\n"
    "  --tile               Enable shared-memory tiling (OPT3a)\n"
    "  --gpuField            Generate scalar field on GPU (OPT3b)\n"
    "\n"
    "Field modes:\n"
    "  --plane              Use plane field instead of noise terrain\n"
    "  --planeY V            Plane Y offset (default: 0.0)\n"
    "\n"
    "Grid size:\n"
    "  --Nx N --Ny N --Nz N  Requested grid size.\n"
    "\n"
    "NOTES:\n"
    "  - When --gpuField is OFF and --plane is OFF, the program generates the terrain\n"
    "    with the built-in multi-step expand_and_noise_3D pipeline, which determines\n"
    "    Nx/Ny/Nz automatically (grid size flags are ignored in this mode).\n"
    "  - When --plane is ON, the plane field uses a fixed internal resolution.\n"
    "  - When --gpuField is ON, the scalar field uses a fixed internal resolution.\n";

}

static inline int requireValue(int argc, char** argv, int i, const char* flag) {
    if (i + 1 >= argc) {
        std::cerr << "Missing value after " << flag << "\n";
        std::exit(1);
    }
    return i + 1;
}

static inline Options parseArgs(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];

        if (a == "--help" || a == "-h") {
            printHelp(argv[0]);
            std::exit(0);
        } else if (a == "--output") {
            i = requireValue(argc, argv, i, "--output");
            opt.output = argv[i];
        } else if (a == "--input") {
            i = requireValue(argc, argv, i, "--input");
            opt.input = argv[i];
        } else if (a == "--iso") {
            i = requireValue(argc, argv, i, "--iso");
            opt.iso = std::stof(argv[i]);
        } else if (a == "--Nx") {
            i = requireValue(argc, argv, i, "--Nx");
            opt.Nx = std::stoi(argv[i]);
        } else if (a == "--Ny") {
            i = requireValue(argc, argv, i, "--Ny");
            opt.Ny = std::stoi(argv[i]);
        } else if (a == "--Nz") {
            i = requireValue(argc, argv, i, "--Nz");
            opt.Nz = std::stoi(argv[i]);
        } else if (a == "--plane") {
            opt.planeMode = true;
        } else if (a == "--planeY") {
            i = requireValue(argc, argv, i, "--planeY");
            opt.planeY = std::stof(argv[i]);
        } else if (a == "--seed") {
            i = requireValue(argc, argv, i, "--seed");
            opt.seed = (unsigned)std::stoul(argv[i]);
        } else if (a == "--warmup") {
            i = requireValue(argc, argv, i, "--warmup");
            opt.warmup = std::stoi(argv[i]);
        } else if (a == "--iters") {
            i = requireValue(argc, argv, i, "--iters");
            opt.iters = std::stoi(argv[i]);
        } else if (a == "--chunks") {
            opt.useChunks = true;
        } else if (a == "--cull") {
            opt.cullChunks = true;
        } else if (a == "--tiled") {
            opt.tiled = true;
        } else if (a == "--deterministic") opt.deterministic = true;
        else if (a == "--gpuField")      opt.gpuField = true;
        else {
            std::cerr << "Unknown flag: " << a << "\n";
            std::cerr << "Run --help\n";
            std::exit(1);
        }
    }

    // light validation
    if (opt.cullChunks && !opt.useChunks) {
        std::cerr << "--cull requires --chunks\n";
        std::exit(1);
    }
    if (opt.tiled && !opt.useChunks) {
        std::cerr << "--tiled requires --chunks\n";
        std::exit(1);
    }
    if (opt.Nx < 2 || opt.Ny < 2 || opt.Nz < 2) {
        std::cerr << "Nx/Ny/Nz must be >= 2\n";
        std::exit(1);
    }
    return opt;
}
