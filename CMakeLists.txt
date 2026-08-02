cmake_minimum_required(VERSION 3.20)
project(DepthPeelingRenderer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(tinyobjloader REQUIRED)
find_package(imgui REQUIRED)

# add_subdirectory(external/glfw)
add_subdirectory(vk-bootstrap)

file(GLOB_RECURSE SOURCES src/*.cpp src/*.h)
add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
    src/
    external/
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    Vulkan::Vulkan
    glfw
    vk-bootstrap::vk-bootstrap
    glm::glm
    tinyobjloader::tinyobjloader
    imgui::imgui
)

# ─── Shader compilation (recursive, supports nested dirs) ───

# Collect shaders from both shaders/ and assets/ (recursively)
file(GLOB_RECURSE SHADER_FILES
    shaders/*.vert
    shaders/*.frag
    shaders/*.comp
)

set(SPIRV_OUTPUTS "")

foreach(SHADER ${SHADER_FILES})
    # Get path relative to the source directory to preserve folder structure
    file(RELATIVE_PATH SHADER_REL "${CMAKE_SOURCE_DIR}" "${SHADER}")

    # Output: build/compiled_shaders/<relative_path>.spv
    set(SPIRV_OUTPUT "${CMAKE_BINARY_DIR}/${SHADER_REL}.spv")

    # Ensure the output subdirectory exists
    get_filename_component(SPIRV_OUTPUT_DIR "${SPIRV_OUTPUT}" DIRECTORY)

    add_custom_command(
        OUTPUT "${SPIRV_OUTPUT}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${SPIRV_OUTPUT_DIR}"
        COMMAND Vulkan::glslangValidator -V "${SHADER}" -o "${SPIRV_OUTPUT}"
        DEPENDS "${SHADER}"
        COMMENT "Compiling shader: ${SHADER_REL} -> SPIR-V"
    )
    list(APPEND SPIRV_OUTPUTS "${SPIRV_OUTPUT}")
endforeach()

add_custom_target(compile_shaders ALL DEPENDS ${SPIRV_OUTPUTS})
add_dependencies(${PROJECT_NAME} compile_shaders)

file(GLOB_RECURSE ASSET_FILES "${CMAKE_SOURCE_DIR}/assets/*")

add_custom_target(copy_assets ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/assets"
        "${CMAKE_BINARY_DIR}/assets"
    DEPENDS ${ASSET_FILES}
    COMMENT "Copying assets/ to build directory"
)

add_dependencies(${PROJECT_NAME} copy_assets)