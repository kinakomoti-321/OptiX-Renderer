#pragma once
#define TINYOBJLOADER_IMPLEMENTATION
#include <iostream>
#include <vector>
#include <sutil/sutil.h>
#include <external/tinyobjloader/tiny_obj_loader.h>
#include <myPathTracer/material.h>

int loadTexture(std::vector<std::shared_ptr<Texture>>& textures, std::map<std::string, int>& known_tex, const std::string& filename, const std::string& modelpath,const std::string& tex_type) {
    if (filename == "") return -1;

    if (known_tex.find(filename) != known_tex.end()) {
        return known_tex[filename];
    }
    
    int textureID = (int)textures.size();
    textures.push_back(std::make_shared<Texture>(modelpath +"/"+ filename,tex_type));
    known_tex[filename] = textureID;

    return textureID;
}

inline bool objLoader(
    const std::string& filepath,
	const std::string& filename,
    SceneData& scenedata
	){

    tinyobj::ObjReader reader;
    std::cout << std::endl << "-------------------" << std::endl;
    std::cout << filename << " load start" << std::endl;
    std::cout << "-------------------" << std::endl;

    if (!reader.ParseFromFile(filepath + "/" + filename))
    {
        if (!reader.Error().empty())
        {
            std::cerr << reader.Error();
        }
        return false;
    }

    if (!reader.Warning().empty())
    {

        std::cout << std::endl << "...loading error" << std::endl;
        std::cout << reader.Warning();

    }

    const auto& attrib = reader.GetAttrib();
    const auto& shapes = reader.GetShapes();
    const auto& load_materials = reader.GetMaterials();
    size_t index_offset_ = 0;
    
    bool has_material = load_materials.size() > 0;
    
    std::map<std::string, int> known_tex;

    if (has_material) {
        for (int i = 0; i < load_materials.size(); i++) {
            //Matrial setting
            Material mat;
            mat.material_name = load_materials[i].name;

            //Diffuse Color
            mat.base_color = { load_materials[i].diffuse[0],load_materials[i].diffuse[1],load_materials[i].diffuse[2] };
            mat.base_color_tex = loadTexture(scenedata.textures, known_tex, load_materials[i].diffuse_texname,filepath,"Diffuse");

            //Metallic
            mat.metallic = load_materials[i].metallic;
            mat.metallic_tex = loadTexture(scenedata.textures, known_tex, load_materials[i].metallic_texname,filepath,"Metallic");

            //Roghness
            mat.roughness = load_materials[i].roughness;
            //mat.roughness = 0.5f;
            mat.roughness_tex = loadTexture(scenedata.textures, known_tex, load_materials[i].roughness_texname,filepath,"Roughness");

            //Sheen
            mat.sheen = load_materials[i].sheen;
            mat.sheen_tex = loadTexture(scenedata.textures, known_tex, load_materials[i].sheen_texname,filepath,"Sheen");

            //IOR
            mat.ior = load_materials[i].ior;

            //Specular
            mat.specular = { load_materials[i].specular[0],load_materials[i].specular[1],load_materials[i].specular[2] };
            mat.specular_tex = loadTexture(scenedata.textures, known_tex, load_materials[i].specular_texname,filepath,"Specular");

            //Normal map
            //mat.normal_tex = loadTexture(scenedata.textures, known_tex, load_materials[i].normal_texname,filepath,"Normalmap");
            mat.normal_tex = loadTexture(scenedata.textures, known_tex, load_materials[i].bump_texname,filepath,"Normalmap");

            //Bump map
            mat.bump_tex = loadTexture(scenedata.textures, known_tex, load_materials[i].bump_texname,filepath,"Bumpmap");
            
            //Emmision
            mat.emmision_color = { load_materials[i].emission[0],load_materials[i].emission[1],load_materials[i].emission[2] };
            mat.emmision_color_tex = -1;
            
            Log::DebugLog(mat);
            scenedata.material.push_back(mat);
        }
    }
    else {
        Material mat;
        mat.base_color = { 1,1,1 };
        mat.emmision_color = { 0,0,0 };
        scenedata.material.push_back(mat);
    }
    

    //シェイプ数分のループ
    for (size_t s = 0; s < shapes.size(); s++) {
        size_t index_offset = 0;
        //シェイプのフェイス分のループ
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            //シェイプsの面fに含まれる頂点数
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);
            float3 nv[3];
            for (size_t v = 0; v < fv; v++) {
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                //シェイプのv番目の頂点座標
                tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];

                scenedata.vertices.push_back({ vx ,vy, vz});

                if (idx.normal_index >= 0) {
                    //シェイプのv番目の法線ベクトル
                    tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                    tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                    tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];

                    scenedata.normal.push_back({ nx,ny,nz });
                }
                else {
                    nv[v] = float3{ vx, vy, vz };
                }
                if (idx.texcoord_index >= 0) {
                    //シェイプのv番目のUV
                    tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                    tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];

                    // std::cout << f << std::endl;
                    scenedata.uv.push_back({ tx, ty});
                }
                else {
                    scenedata.uv.push_back({ 0, 0});
                }
                //v番目のindex
                scenedata.index.push_back(static_cast<unsigned int>(index_offset_ + index_offset + v));
            }
            
            if (attrib.normals.size() == 0) {
                const float3 nv1 = normalize(nv[1] - nv[0]);
                const float3 nv2 = normalize(nv[2] - nv[0]);
                float3 geoNormal = normalize(cross(nv1, nv2));
                scenedata.normal.push_back(geoNormal);
                scenedata.normal.push_back(geoNormal);
                scenedata.normal.push_back(geoNormal);
            }

            if (has_material) {
                scenedata.material_index.push_back(shapes[s].mesh.material_ids[f]);
            }
            else {
                scenedata.material_index.push_back(0);
            }
            index_offset += fv;

        }
        index_offset_ += index_offset;
    }
    
    
    
    std::cout << std::endl << "...model information" << std::endl;
    std::cout << "the number of vertex : " << scenedata.vertices.size() << std::endl;
    std::cout << "the number of normal : " << scenedata.normal.size() << std::endl;
    std::cout << "the number of texcoord : " << scenedata.uv.size() << std::endl;
    std::cout << "the number of face : " << scenedata.index.size() << std::endl;
    std::cout << "the number of material : " << load_materials.size() << std::endl;
    return true;
}

