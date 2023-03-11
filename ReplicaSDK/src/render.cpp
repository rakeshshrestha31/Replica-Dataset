// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#include <EGL.h>
#include <PTexLib.h>
#include <pangolin/image/image_convert.h>

#include "GLCheck.h"
#include "MirrorRenderer.h"

#include <fstream>

Eigen::Matrix4d load_extrinsic(char filename[]) {
    std::ifstream f(filename);
    double x[16];
    int i = 0;
    while (f >> x[i++]) {}

    Eigen::Matrix4d m= Eigen::Map<Eigen::Matrix4d>(x, 4, 4);
    return m.transpose();
}

int main(int argc, char* argv[]) {
  ASSERT(argc == 3 || argc == 4, "Usage: ./ReplicaRenderer mesh.ply /path/to/atlases [mirrorFile]");

  const std::string meshFile(argv[1]);
  const std::string atlasFolder(argv[2]);
  ASSERT(pangolin::FileExists(meshFile));
  ASSERT(pangolin::FileExists(atlasFolder));

  std::string surfaceFile;
  if (argc == 4) {
    surfaceFile = std::string(argv[3]);
    ASSERT(pangolin::FileExists(surfaceFile));
  }

  const int width = 1280;
  const int height = 960;
  bool renderDepth = true;
  float depthScale = 65535.0f * 0.1f;

  auto fx = width / 2.0f;
  auto fy = width / 2.0f;
  auto cx = (width - 1.0f) / 2.0f;
  auto cy = (height - 1.0f) / 2.0f;
  auto near = 0.1f;
  auto far = 100.0f;

  Eigen::IOFormat matrix_fmt(
      Eigen::StreamPrecision, Eigen::DontAlignCols, " ", "\n", "", ""
  );

  // Setup EGL
  EGLCtx egl;

  egl.PrintInformation();

  if(!checkGLVersion()) {
    return 1;
  }

  //Don't draw backfaces
  const GLenum frontFace = GL_CCW;
  glFrontFace(frontFace);

  // Setup a framebuffer
  pangolin::GlTexture render(width, height);
  pangolin::GlRenderBuffer renderBuffer(width, height);
  pangolin::GlFramebuffer frameBuffer(render, renderBuffer);

  pangolin::GlTexture depthTexture(width, height, GL_R32F, false, 0, GL_RED, GL_FLOAT, 0);
  pangolin::GlFramebuffer depthFrameBuffer(depthTexture, renderBuffer);

  pangolin::OpenGlMatrixSpec proj_mat = pangolin::ProjectionMatrixRDF_BottomLeft(
      width,
      height,
      fx, fy,
      cx, cy,
      near, far);

  char filename[1000] = "projection.txt";
  {
    std::ofstream projection_file(filename);
    projection_file << Eigen::Matrix4d(proj_mat).format(matrix_fmt);
  }

  sprintf(filename, "intrinsics.txt");
  {
    std::ofstream intrinsics_file(filename);
    intrinsics_file
        << fx << " " << fy << " "
        << cx << " " << cy << " "
        << width << " " << height << " "
        << near << " " << far;
  }

  // Setup a camera
  pangolin::OpenGlRenderState s_cam(
      proj_mat,
      pangolin::ModelViewLookAtRDF(0, 0, 4, 0, 0, 0, 0, 1, 0));

  // Start at some origin
  Eigen::Matrix4d T_camera_world = s_cam.GetModelViewMatrix();

  // And move to the left
  Eigen::Matrix4d T_new_old = Eigen::Matrix4d::Identity();

  T_new_old.topRightCorner(3, 1) = Eigen::Vector3d(0.025, 0, 0);

  // load mirrors
  std::vector<MirrorSurface> mirrors;
  if (surfaceFile.length()) {
    std::ifstream file(surfaceFile);
    picojson::value json;
    picojson::parse(json, file);

    for (size_t i = 0; i < json.size(); i++) {
      mirrors.emplace_back(json[i]);
    }
    std::cout << "Loaded " << mirrors.size() << " mirrors" << std::endl;
  }

  const std::string shadir = STR(SHADER_DIR);
  MirrorRenderer mirrorRenderer(mirrors, width, height, shadir);

  // load mesh and textures
  PTexMesh ptexMesh(meshFile, atlasFolder);

  pangolin::ManagedImage<Eigen::Matrix<uint8_t, 3, 1>> image(width, height);
  pangolin::ManagedImage<float> depthImage(width, height);
  pangolin::ManagedImage<uint16_t> depthImageInt(width, height);

  // Render some frames
  const size_t numFrames = 75;
  for (size_t i = 0; i < numFrames; i++) {
    std::cout << "\rRendering frame " << i + 1 << "/" << numFrames << "... ";
    std::cout.flush();

    snprintf(filename, 1000, "/tmp/extrinsic_%06zu.txt", i);
    s_cam.GetModelViewMatrix() = load_extrinsic(filename);

    // Render
    frameBuffer.Bind();
    glPushAttrib(GL_VIEWPORT_BIT);
    glViewport(0, 0, width, height);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    glEnable(GL_CULL_FACE);

    ptexMesh.Render(s_cam);

    glDisable(GL_CULL_FACE);

    glPopAttrib(); //GL_VIEWPORT_BIT
    frameBuffer.Unbind();

    for (size_t i = 0; i < mirrors.size(); i++) {
      MirrorSurface& mirror = mirrors[i];
      // capture reflections
      mirrorRenderer.CaptureReflection(mirror, ptexMesh, s_cam, frontFace);

      frameBuffer.Bind();
      glPushAttrib(GL_VIEWPORT_BIT);
      glViewport(0, 0, width, height);

      // render mirror
      mirrorRenderer.Render(mirror, mirrorRenderer.GetMaskTexture(i), s_cam);

      glPopAttrib(); //GL_VIEWPORT_BIT
      frameBuffer.Unbind();
    }

    // Download and save
    render.Download(image.ptr, GL_RGB, GL_UNSIGNED_BYTE);

    snprintf(filename, 1000, "%06zu.jpg", i);

    pangolin::SaveImage(
        image.UnsafeReinterpret<uint8_t>(),
        pangolin::PixelFormatFromString("RGB24"),
        std::string(filename));

    snprintf(filename, 1000, "pose%06zu.txt", i);
    std::ofstream pose_file(filename);
    pose_file << Eigen::Matrix4d(s_cam.GetModelViewMatrix()).format(matrix_fmt);
    pose_file.close();

    // snprintf(filename, 1000, "projection%06zu.txt", i);
    // std::ofstream projection_file(filename);
    // projection_file << Eigen::Matrix4d(s_cam.GetProjectionMatrix()).format(matrix_fmt);
    // projection_file.close();

    if (renderDepth) {
      // render depth
      depthFrameBuffer.Bind();
      glPushAttrib(GL_VIEWPORT_BIT);
      glViewport(0, 0, width, height);
      glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

      glEnable(GL_CULL_FACE);

      ptexMesh.RenderDepth(s_cam, depthScale);

      glDisable(GL_CULL_FACE);

      glPopAttrib(); //GL_VIEWPORT_BIT
      depthFrameBuffer.Unbind();

      depthTexture.Download(depthImage.ptr, GL_RED, GL_FLOAT);

      // convert to 16-bit int
      for(size_t i = 0; i < depthImage.Area(); i++)
          depthImageInt[i] = static_cast<uint16_t>(depthImage[i] + 0.5f);

      snprintf(filename, 1000, "depth%06zu.png", i);
      pangolin::SaveImage(
          depthImageInt.UnsafeReinterpret<uint8_t>(),
          pangolin::PixelFormatFromString("GRAY16LE"),
          std::string(filename), true, 34.0f);
    }

    // Move the camera
    // T_camera_world = T_camera_world * T_new_old.inverse();

    // s_cam.GetModelViewMatrix() = T_camera_world;
  }
  std::cout << "\rRendering frame " << numFrames << "/" << numFrames << "... done" << std::endl;

  return 0;
}

