#pragma once

#include "mesh.h"
#include <string>
#include <vector>
#include <glm/glm.hpp>

class Scene {
public:
    bool loadFromOBJ(const std::string& filepath);

    const std::vector<Vertex>& getVertices() const { return m_vertices; }
    const std::vector<uint32_t>& getIndices() const { return m_indices; }
    const std::vector<SceneObject>& getObjects() const { return m_objects; }
    const std::vector<MaterialInfo>& getMaterials() const { return m_materials; }

    // Returns all unique texture filenames (full paths) referenced by materials
    std::vector<std::string> getTextureFilenames() const;

    uint32_t getTotalVertexCount() const { return static_cast<uint32_t>(m_vertices.size()); }
    uint32_t getTotalIndexCount() const { return static_cast<uint32_t>(m_indices.size()); }
    uint32_t getObjectCount() const { return static_cast<uint32_t>(m_objects.size()); }

    glm::vec3 getBoundsMin() const { return m_boundsMin; }
    glm::vec3 getBoundsMax() const { return m_boundsMax; }
    glm::vec3 getCenter() const { return (m_boundsMin + m_boundsMax) * 0.5f; }
    float getRadius() const;

    // Call after textures are loaded to assign texture indices to objects
    void resolveTextureIndices(const std::vector<std::string>& loadedTexturePaths);

private:
    std::vector<Vertex> m_vertices;
    std::vector<uint32_t> m_indices;
    std::vector<SceneObject> m_objects;
    std::vector<MaterialInfo> m_materials;

    std::string m_baseDir; // Directory of the OBJ file (for resolving texture paths)

    glm::vec3 m_boundsMin = glm::vec3(std::numeric_limits<float>::max());
    glm::vec3 m_boundsMax = glm::vec3(std::numeric_limits<float>::lowest());

    void computeBounds();
};