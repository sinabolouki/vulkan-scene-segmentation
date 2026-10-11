#include "application.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <system_error>

namespace
{
constexpr std::string_view DEFAULT_SCENE_PATH = "assets/simple_room/simple_room.obj";
    // std::string scenePath = "../assets/circles16.obj";
    // std::string scenePath = "../assets/simple_scene.obj";
// constexpr std::string_view DEFAULT_SCENE_PATH = "assets/sponza/sponza.obj";
    // std::string scenePath = "assets/office/model.obj";

void printUsage(const char *program)
{
    std::cout << "Usage: " << program << " [--scene <path/to/scene.obj>] [--help]\n"
              << "  --scene <path>  Load an OBJ scene (absolute or relative to the working directory).\n"
              << "  --help, -h      Show this help.\n"
              << "Default scene: " << DEFAULT_SCENE_PATH << '\n';
}
}

int main(int argc, char *argv[])
{
    std::string scenePath(DEFAULT_SCENE_PATH);

    for (int i = 1; i < argc; ++i)
    {
        const std::string_view arg(argv[i]);
        if (arg == "--help" || arg == "-h")
        {
            printUsage(argv[0]);
            return EXIT_SUCCESS;
        }
        if (arg != "--scene")
        {
            std::cerr << "Unknown argument: " << arg << '\n';
            printUsage(argv[0]);
            return EXIT_FAILURE;
        }
        if (i + 1 >= argc || std::string_view(argv[i + 1]).empty()
            || std::string_view(argv[i + 1]).starts_with('-'))
        {
            std::cerr << "--scene requires a file path.\n";
            printUsage(argv[0]);
            return EXIT_FAILURE;
        }
        scenePath = argv[++i];
    }

    // Validate before creating a window or initializing Vulkan. Passing an
    // absolute path also lets Scene resolve materials and textures beside it.
    std::error_code error;
    const auto resolvedPath = std::filesystem::absolute(scenePath, error);
    if (error)
    {
        std::cerr << "[App] Cannot resolve scene path: " << scenePath
                  << " (" << error.message() << ")\n";
        return EXIT_FAILURE;
    }
    if (!std::filesystem::is_regular_file(resolvedPath, error))
    {
        std::cerr << "[App] Scene file not found or not a regular file: " << resolvedPath << '\n';
        if (error)
            std::cerr << "[App] " << error.message() << '\n';
        std::cerr << "Use --scene \"path/to/scene.obj\" to select an existing OBJ file.\n";
        return EXIT_FAILURE;
    }

    std::cout << "[App] Scene: " << resolvedPath << '\n';
    Application app;

    if (!app.init(1280, 720, "Depth Peeling Renderer", resolvedPath.string()))
    {
        return EXIT_FAILURE;
    }

    app.run();
    app.cleanup();

    return EXIT_SUCCESS;
}
