#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include "scene.h"
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

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
    // Store base directory for resolving texture paths
    m_baseDir = fs::path(filepath).parent_path().string();
    if (!m_baseDir.empty() && m_baseDir.back() != '/' && m_baseDir.back() != '\\') {
        m_baseDir += '/';
    }

    tinyobj::ObjReader reader;
    tinyobj::ObjReaderConfig config;
    config.triangulate = true;
    config.vertex_color = true;
    config.mtl_search_path = m_baseDir; // Look for MTL in same directory as OBJ

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
    const auto& materials = reader.GetMaterials();

    std::cout << "[Scene] Loading: " << filepath << std::endl;
    std::cout << "[Scene] Base dir: " << m_baseDir << std::endl;
    std::cout << "[Scene] Vertices: " << attrib.vertices.size() / 3 << std::endl;
    std::cout << "[Scene] Normals:  " << attrib.normals.size() / 3 << std::endl;
    std::cout << "[Scene] UVs:      " << attrib.texcoords.size() / 2 << std::endl;
    std::cout << "[Scene] Materials: " << materials.size() << std::endl;
    std::cout << "[Scene] Objects:  " << shapes.size() << std::endl;

    // ─── Parse Materials ─────────────────────────────────────────────────────
    m_materials.resize(materials.size());
    for (size_t i = 0; i < materials.size(); i++) {
        m_materials[i].name = materials[i].name;
        m_materials[i].diffuseColor = glm::vec3(
            materials[i].diffuse[0],
            materials[i].diffuse[1],
            materials[i].diffuse[2]
        );

        // Resolve texture path
        if (!materials[i].diffuse_texname.empty()) {
            std::string texPath = m_baseDir + materials[i].diffuse_texname;
            // Normalize path separators
            std::replace(texPath.begin(), texPath.end(), '\\', '/');
            m_materials[i].diffuseTexture = texPath;
        }

        std::cout << "[Scene]   Material " << i << ": \"" << m_materials[i].name
                  << "\" Kd=(" << m_materials[i].diffuseColor.r << ", "
                  << m_materials[i].diffuseColor.g << ", "
                  << m_materials[i].diffuseColor.b << ")";
        if (!m_materials[i].diffuseTexture.empty()) {
            std::cout << " tex=\"" << materials[i].diffuse_texname << "\"";
        }
        std::cout << std::endl;
    }

    // ─── Process Geometry ────────────────────────────────────────────────────
    std::unordered_map<Vertex, uint32_t, VertexHash> uniqueVertices;

    for (size_t shapeIdx = 0; shapeIdx < shapes.size(); shapeIdx++) {
        const auto& shape = shapes[shapeIdx];

        SceneObject obj;
        obj.name = shape.name;
        obj.objectID = static_cast<uint32_t>(shapeIdx);
        obj.indexOffset = static_cast<uint32_t>(m_indices.size());

        // Determine dominant material for this shape
        // (Sponza typically has one material per shape)
        if (!shape.mesh.material_ids.empty()) {
            // Use the first face's material as the object's material
            int matId = shape.mesh.material_ids[0];
            if (matId >= 0 && matId < (int)m_materials.size()) {
                obj.materialIndex = matId;
                obj.diffuseColor = m_materials[matId].diffuseColor;
            }
        }

        uint32_t indexCountForObject = 0;
        size_t indexOffset = 0;

        for (size_t faceIdx = 0; faceIdx < shape.mesh.num_face_vertices.size(); faceIdx++) {
            uint32_t faceVerts = shape.mesh.num_face_vertices[faceIdx];

            // Get per-face material (may differ from shape default)
            int faceMaterialId = -1;
            if (faceIdx < shape.mesh.material_ids.size()) {
                faceMaterialId = shape.mesh.material_ids[faceIdx];
            }

            glm::vec3 faceDiffuse = glm::vec3(1.0f);
            if (faceMaterialId >= 0 && faceMaterialId < (int)m_materials.size()) {
                faceDiffuse = m_materials[faceMaterialId].diffuseColor;
            }

            for (uint32_t v = 0; v < faceVerts; v++) {
                tinyobj::index_t idx = shape.mesh.indices[indexOffset + v];

                Vertex vertex{};

                // Position
                vertex.position = {
                    attrib.vertices[3 * idx.vertex_index + 0],
                    attrib.vertices[3 * idx.vertex_index + 1],
                    attrib.vertices[3 * idx.vertex_index + 2]
                };

                // Normal
                if (idx.normal_index >= 0 && !attrib.normals.empty()) {
                    vertex.normal = {
                        attrib.normals[3 * idx.normal_index + 0],
                        attrib.normals[3 * idx.normal_index + 1],
                        attrib.normals[3 * idx.normal_index + 2]
                    };
                } else {
                    vertex.normal = {0.0f, 1.0f, 0.0f};
                }

                // Color: use material diffuse color (baked into vertex)
                vertex.color = faceDiffuse;

                // UV
                if (idx.texcoord_index >= 0 && !attrib.texcoords.empty()) {
                    vertex.uv = {
                        attrib.texcoords[2 * idx.texcoord_index + 0],
                        1.0f - attrib.texcoords[2 * idx.texcoord_index + 1]
                    };
                } else {
                    vertex.uv = {0.0f, 0.0f};
                }

                // Deduplicate
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
                  << "\" - " << obj.indexCount / 3 << " tris"
                  << " (mat: " << obj.materialIndex << ")" << std::endl;
    }

    computeBounds();

    std::cout << "[Scene] Total: " << m_vertices.size() << " unique vertices, "
              << m_indices.size() << " indices, "
              << m_objects.size() << " objects." << std::endl;

    return true;
}

std::vector<std::string> Scene::getTextureFilenames() const {
    std::vector<std::string> filenames;
    for (const auto& mat : m_materials) {
        if (!mat.diffuseTexture.empty()) {
            // Avoid duplicates
            if (std::find(filenames.begin(), filenames.end(), mat.diffuseTexture) == filenames.end()) {
                filenames.push_back(mat.diffuseTexture);
            }
        }
    }
    return filenames;
}

void Scene::resolveTextureIndices(const std::vector<std::string>& loadedTexturePaths) {
    // Assign texture indices to materials
    for (auto& mat : m_materials) {
        if (mat.diffuseTexture.empty()) {
            mat.textureIndex = -1;
            continue;
        }
        auto it = std::find(loadedTexturePaths.begin(), loadedTexturePaths.end(), mat.diffuseTexture);
        if (it != loadedTexturePaths.end()) {
            mat.textureIndex = static_cast<int32_t>(std::distance(loadedTexturePaths.begin(), it));
        } else {
            mat.textureIndex = -1;
        }
    }

    // Propagate to objects
    for (auto& obj : m_objects) {
        if (obj.materialIndex >= 0 && obj.materialIndex < (int)m_materials.size()) {
            obj.textureIndex = m_materials[obj.materialIndex].textureIndex;
        }
    }
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
    std::cout << "[Scene] Bounds: min(" << m_boundsMin.x << ", " << m_boundsMin.y << ", " << m_boundsMin.z
              << ") max(" << m_boundsMax.x << ", " << m_boundsMax.y << ", " << m_boundsMax.z << ")" << std::endl;
    return glm::length(m_boundsMax - m_boundsMin) * 0.5f;
}