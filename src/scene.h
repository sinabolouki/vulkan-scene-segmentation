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

    uint32_t getTotalVertexCount() const { return static_cast<uint32_t>(m_vertices.size()); }
    uint32_t getTotalIndexCount() const { return static_cast<uint32_t>(m_indices.size()); }
    uint32_t getObjectCount() const { return static_cast<uint32_t>(m_objects.size()); }

    // Bounding box for camera setup
    glm::vec3 getBoundsMin() const { return m_boundsMin; }
    glm::vec3 getBoundsMax() const { return m_boundsMax; }
    glm::vec3 getCenter() const { return (m_boundsMin + m_boundsMax) * 0.5f; }
    float getRadius() const;

private:
    std::vector<Vertex> m_vertices;
    std::vector<uint32_t> m_indices;
    std::vector<SceneObject> m_objects;

    glm::vec3 m_boundsMin = glm::vec3(std::numeric_limits<float>::max());
    glm::vec3 m_boundsMax = glm::vec3(std::numeric_limits<float>::lowest());

    void computeBounds();
};