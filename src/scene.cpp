#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include "scene.h"
#include <iostream>
#include <unordered_map>

// Hash for deduplicating vertices
struct VertexHash {
    size_t operator()(const Vertex& v) const {
        size_t h = 0;
        auto hashFloat = [&](float f) {
            h ^= std::hash<float>{}(f) + 0x9e3779b9 + (h << 6) + (h >> 2);
        };
        hashFloat(v.position.x); hashFloat(v.position.y); hashFloat(v.position.z);
        hashFloat(v.normal.x);   hashFloat(v.normal.y);   hashFloat(v.normal.z);
        hashFloat(v.color.r);    hashFloat(v.color.g);    hashFloat(v.color.b);
        hashFloat(v.uv.x);       hashFloat(v.uv.y);
        return h;
    }
};

static bool operator==(const Vertex& a, const Vertex& b) {
    return a.position == b.position &&
           a.normal == b.normal &&
           a.color == b.color &&
           a.uv == b.uv;
}

bool Scene::loadFromOBJ(const std::string& filepath) {
    tinyobj::ObjReader reader;
    tinyobj::ObjReaderConfig config;
    config.triangulate = true;         // Ensure all faces are triangles
    config.vertex_color = true;        // Enable vertex color parsing

    if (!reader.ParseFromFile(filepath, config)) {
        if (!reader.Error().empty()) {
            std::cerr << "[Scene] TinyOBJ Error: " << reader.Error() << std::endl;
        }
        return false;
    }

    if (!reader.Warning().empty()) {
        std::cout << "[Scene] TinyOBJ Warning: " << reader.Warning() << std::endl;
    }

    const auto& attrib = reader.GetAttrib();
    const auto& shapes = reader.GetShapes();

    std::cout << "[Scene] Loading: " << filepath << std::endl;
    std::cout << "[Scene] Vertices: " << attrib.vertices.size() / 3 << std::endl;
    std::cout << "[Scene] Normals:  " << attrib.normals.size() / 3 << std::endl;
    std::cout << "[Scene] Colors:   " << attrib.colors.size() / 3 << std::endl;
    std::cout << "[Scene] Objects:  " << shapes.size() << std::endl;

    // Use a global vertex deduplication map
    std::unordered_map<Vertex, uint32_t, VertexHash> uniqueVertices;

    // Process each shape (object) in the OBJ file
    for (size_t shapeIdx = 0; shapeIdx < shapes.size(); shapeIdx++) {
        const auto& shape = shapes[shapeIdx];

        SceneObject obj;
        obj.name = shape.name;
        obj.objectID = static_cast<uint32_t>(shapeIdx);
        obj.indexOffset = static_cast<uint32_t>(m_indices.size());

        uint32_t indexCountForObject = 0;

        // Iterate over faces
        size_t indexOffset = 0;
        for (size_t faceIdx = 0; faceIdx < shape.mesh.num_face_vertices.size(); faceIdx++) {
            uint32_t faceVerts = shape.mesh.num_face_vertices[faceIdx];

            for (uint32_t v = 0; v < faceVerts; v++) {
                tinyobj::index_t idx = shape.mesh.indices[indexOffset + v];

                Vertex vertex{};

                // Position (always present)
                vertex.position = {
                    attrib.vertices[3 * idx.vertex_index + 0],
                    attrib.vertices[3 * idx.vertex_index + 1],
                    attrib.vertices[3 * idx.vertex_index + 2]
                };

                // Normal (may not be present)
                if (idx.normal_index >= 0 && !attrib.normals.empty()) {
                    vertex.normal = {
                        attrib.normals[3 * idx.normal_index + 0],
                        attrib.normals[3 * idx.normal_index + 1],
                        attrib.normals[3 * idx.normal_index + 2]
                    };
                } else {
                    vertex.normal = {0.0f, 1.0f, 0.0f}; // Default up
                }

                // Vertex Color (stored per vertex_index in tinyobj)
                // tinyobj stores colors in attrib.colors, indexed by vertex_index
                if (!attrib.colors.empty() && (3 * idx.vertex_index + 2) < attrib.colors.size()) {
                    vertex.color = {
                        attrib.colors[3 * idx.vertex_index + 0],
                        attrib.colors[3 * idx.vertex_index + 1],
                        attrib.colors[3 * idx.vertex_index + 2]
                    };
                } else {
                    vertex.color = {1.0f, 1.0f, 1.0f}; // Default white
                }

                // UV / Texture coordinates (may not be present)
                if (idx.texcoord_index >= 0 && !attrib.texcoords.empty()) {
                    vertex.uv = {
                        attrib.texcoords[2 * idx.texcoord_index + 0],
                        1.0f - attrib.texcoords[2 * idx.texcoord_index + 1] // Flip V for Vulkan
                    };
                } else {
                    vertex.uv = {0.0f, 0.0f};
                }

                // Deduplicate vertices
                if (uniqueVertices.count(vertex) == 0) {
                    uniqueVertices[vertex] = static_cast<uint32_t>(m_vertices.size());
                    m_vertices.push_back(vertex);
                }

                m_indices.push_back(uniqueVertices[vertex]);
                indexCountForObject++;
            }

            indexOffset += faceVerts;
        }

        obj.indexCount = indexCountForObject;
        m_objects.push_back(obj);

        std::cout << "[Scene]   Object " << shapeIdx << ": \"" << obj.name
                  << "\" - " << obj.indexCount / 3 << " triangles" << std::endl;
    }

    computeBounds();

    std::cout << "[Scene] Total: " << m_vertices.size() << " unique vertices, "
              << m_indices.size() << " indices, "
              << m_objects.size() << " objects." << std::endl;
    std::cout << "[Scene] Bounds: [" << m_boundsMin.x << ", " << m_boundsMin.y << ", " << m_boundsMin.z
              << "] to [" << m_boundsMax.x << ", " << m_boundsMax.y << ", " << m_boundsMax.z << "]" << std::endl;

    return true;
}

void Scene::computeBounds() {
    m_boundsMin = glm::vec3(std::numeric_limits<float>::max());
    m_boundsMax = glm::vec3(std::numeric_limits<float>::lowest());

    for (const auto& v : m_vertices) {
        m_boundsMin = glm::min(m_boundsMin, v.position);
        m_boundsMax = glm::max(m_boundsMax, v.position);
    }
}

float Scene::getRadius() const {
    return glm::length(m_boundsMax - m_boundsMin) * 0.5f;
}