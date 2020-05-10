#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

#include <QtGui/QKeyEvent>

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Tools/Smoother/JacobiLaplaceSmootherT.hh>
#include "Eigen/Dense"
using namespace Eigen;

//#define BETTER_MEAN_CURVATURE

#ifdef BETTER_MEAN_CURVATURE
#include "Eigen/Eigenvalues"
#include "Eigen/Geometry"
#include "Eigen/LU"
#include "Eigen/SVD"
#endif

#include "MyViewer.h"

#ifdef _WIN32
#define GL_CLAMP_TO_EDGE 0x812F
#define GL_BGRA 0x80E1
#endif

MyViewer::MyViewer(QWidget *parent) :
  QGLViewer(parent), model_type(ModelType::NONE),
  mean_min(0.0), mean_max(0.0), cutoff_ratio(0.05), mid_insert(false),
  show_control_points(true), show_solid(true), show_wireframe(false),keep_surface(false),
  visualization(Visualization::PLAIN), slicing_dir(0, 0, 1), slicing_scaling(1)
{
  setSelectRegionWidth(10);
  setSelectRegionHeight(10);
  axes.shown = false;
}

MyViewer::~MyViewer() {
  glDeleteTextures(1, &isophote_texture);
  glDeleteTextures(1, &slicing_texture);
}

void MyViewer::updateMeanMinMax() {
  size_t n = mesh.n_vertices();
  if (n == 0)
    return;

  std::vector<double> mean;
  mean.reserve(n);
  for (auto v : mesh.vertices())
    mean.push_back(mesh.data(v).mean);

  std::sort(mean.begin(), mean.end());
  size_t k = (double)n * cutoff_ratio;
  mean_min = std::min(mean[k ? k-1 : 0], 0.0);
  mean_max = std::max(mean[n-k], 0.0);
}

void MyViewer::localSystem(const MyViewer::Vector &normal,
                           MyViewer::Vector &u, MyViewer::Vector &v) {
  // Generates an orthogonal (u,v) coordinate system in the plane defined by `normal`.
  int maxi = 0, nexti = 1;
  double max = std::abs(normal[0]), next = std::abs(normal[1]);
  if (max < next) {
    std::swap(max, next);
    std::swap(maxi, nexti);
  }
  if (std::abs(normal[2]) > max) {
    nexti = maxi;
    maxi = 2;
  } else if (std::abs(normal[2]) > next)
    nexti = 2;

  u.vectorize(0.0);
  u[nexti] = -normal[maxi];
  u[maxi] = normal[nexti];
  u /= u.norm();
  v = normal % u;
}

double MyViewer::voronoiWeight(MyViewer::MyMesh::HalfedgeHandle in_he) {
  // Returns the area of the triangle bounded by in_he that is closest
  // to the vertex pointed to by in_he.
  auto next = mesh.next_halfedge_handle(in_he);
  auto prev = mesh.prev_halfedge_handle(in_he);
  double c2 = mesh.calc_edge_vector(in_he).sqrnorm();
  double b2 = mesh.calc_edge_vector(next).sqrnorm();
  double a2 = mesh.calc_edge_vector(prev).sqrnorm();
  double alpha = mesh.calc_sector_angle(in_he);

  if (a2 + b2 < c2)                // obtuse gamma
    return 0.125 * b2 * std::tan(alpha);
  if (a2 + c2 < b2)                // obtuse beta
    return 0.125 * c2 * std::tan(alpha);
  if (b2 + c2 < a2) {              // obtuse alpha
    double b = std::sqrt(b2), c = std::sqrt(c2);
    double total_area = 0.5 * b * c * std::sin(alpha);
    double beta  = mesh.calc_sector_angle(prev);
    double gamma = mesh.calc_sector_angle(next);
    return total_area - 0.125 * (b2 * std::tan(gamma) + c2 * std::tan(beta));
  }

  double r2 = 0.25 * a2 / std::pow(std::sin(alpha), 2); // squared circumradius
  auto area = [r2](double x2) {
    return 0.125 * std::sqrt(x2) * std::sqrt(std::max(4.0 * r2 - x2, 0.0));
  };
  return area(b2) + area(c2);
}

#ifndef BETTER_MEAN_CURVATURE
void MyViewer::updateMeanCurvature(bool update_min_max) {
  std::map<MyMesh::FaceHandle, double> face_area;
  std::map<MyMesh::VertexHandle, double> vertex_area;

  for (auto f : mesh.faces())
    face_area[f] = mesh.calc_sector_area(mesh.halfedge_handle(f));

  // Compute triangle strip areas
  for (auto v : mesh.vertices()) {
    vertex_area[v] = 0;
    mesh.data(v).mean = 0;
    for (auto f : mesh.vf_range(v))
      vertex_area[v] += face_area[f];
    vertex_area[v] /= 3.0;
  }

  // Compute mean values using dihedral angles
  for (auto v : mesh.vertices()) {
    for (auto h : mesh.vih_range(v)) {
      auto vec = mesh.calc_edge_vector(h);
      double angle = mesh.calc_dihedral_angle(h); // signed; returns 0 at the boundary
      mesh.data(v).mean += angle * vec.norm();
    }
    mesh.data(v).mean *= 0.25 / vertex_area[v];
  }

  if (update_min_max)
    updateMeanMinMax();
}
#else // BETTER_MEAN_CURVATURE
void MyViewer::updateMeanCurvature(bool update_min_max) {
  // As in the paper:
  //   S. Rusinkiewicz, Estimating curvatures and their derivatives on triangle meshes.
  //     3D Data Processing, Visualization and Transmission, IEEE, 2004.

  std::map<MyMesh::VertexHandle, Vector> efgp; // 2nd principal form
  std::map<MyMesh::VertexHandle, double> wp;   // accumulated weight

  // Initial setup
  for (auto v : mesh.vertices()) {
    efgp[v].vectorize(0.0);
    wp[v] = 0.0;
  }

  for (auto f : mesh.faces()) {
    // Setup local edges, vertices and normals
    auto h0 = mesh.halfedge_handle(f);
    auto h1 = mesh.next_halfedge_handle(h0);
    auto h2 = mesh.next_halfedge_handle(h1);
    auto e0 = mesh.calc_edge_vector(h0);
    auto e1 = mesh.calc_edge_vector(h1);
    auto e2 = mesh.calc_edge_vector(h2);
    auto n0 = mesh.normal(mesh.to_vertex_handle(h1));
    auto n1 = mesh.normal(mesh.to_vertex_handle(h2));
    auto n2 = mesh.normal(mesh.to_vertex_handle(h0));

    Vector n = mesh.normal(f), u, v;
    localSystem(n, u, v);

    // Solve a LSQ equation for (e,f,g) of the face
    Eigen::MatrixXd A(6, 3);
    A << (e0 | u), (e0 | v),    0.0,
            0.0,   (e0 | u), (e0 | v),
         (e1 | u), (e1 | v),    0.0,
            0.0,   (e1 | u), (e1 | v),
         (e2 | u), (e2 | v),    0.0,
            0.0,   (e2 | u), (e2 | v);
    Eigen::VectorXd b(6);
    b << ((n2 - n1) | u),
         ((n2 - n1) | v),
         ((n0 - n2) | u),
         ((n0 - n2) | v),
         ((n1 - n0) | u),
         ((n1 - n0) | v);
    Eigen::Vector3d x = A.fullPivLu().solve(b);

    Eigen::Matrix2d F;          // Fundamental matrix for the face
    F << x(0), x(1),
         x(1), x(2);

    for (auto h : mesh.fh_range(f)) {
      auto p = mesh.to_vertex_handle(h);

      // Rotate the (up,vp) local coordinate system to be coplanar with that of the face
      Vector np = mesh.normal(p), up, vp;
      localSystem(np, up, vp);
      auto axis = (np % n).normalize();
      double angle = std::acos(std::min(std::max(n | np, -1.0), 1.0));
      auto rotation = Eigen::AngleAxisd(angle, Eigen::Vector3d(axis.data()));
      Eigen::Vector3d up1(up.data()), vp1(vp.data());
      up1 = rotation * up1;    vp1 = rotation * vp1;
      up = Vector(up1.data()); vp = Vector(vp1.data());

      // Compute the vertex-local (e,f,g)
      double e, f, g;
      Eigen::Vector2d upf, vpf;
      upf << (up | u), (up | v);
      vpf << (vp | u), (vp | v);
      e = upf.transpose() * F * upf;
      f = upf.transpose() * F * vpf;
      g = vpf.transpose() * F * vpf;

      // Accumulate the results with Voronoi weights
      double w = voronoiWeight(h);
      efgp[p] += Vector(e, f, g) * w;
      wp[p] += w;
    }
  }

  // Compute the principal curvatures
  for (auto v : mesh.vertices()) {
    auto &efg = efgp[v];
    efg /= wp[v];
    Eigen::Matrix2d F;
    F << efg[0], efg[1],
         efg[1], efg[2];
    auto k = F.eigenvalues();   // always real, because F is a symmetric real matrix
    mesh.data(v).mean = (k(0).real() + k(1).real()) / 2.0;
  }

  if (update_min_max)
    updateMeanMinMax();
}
#endif

static Vec HSV2RGB(Vec hsv) {
  // As in Wikipedia
  double c = hsv[2] * hsv[1];
  double h = hsv[0] / 60;
  double x = c * (1 - std::abs(std::fmod(h, 2) - 1));
  double m = hsv[2] - c;
  Vec rgb(m, m, m);
  if (h <= 1)
    return rgb + Vec(c, x, 0);
  if (h <= 2)
    return rgb + Vec(x, c, 0);
  if (h <= 3)
    return rgb + Vec(0, c, x);
  if (h <= 4)
    return rgb + Vec(0, x, c);
  if (h <= 5)
    return rgb + Vec(x, 0, c);
  if (h <= 6)
    return rgb + Vec(c, 0, x);
  return rgb;
}

Vec MyViewer::meanMapColor(double d) const {
  double red = 0, green = 120, blue = 240; // Hue
  if (d < 0) {
    double alpha = mean_min ? std::min(d / mean_min, 1.0) : 1.0;
    return HSV2RGB({green * (1 - alpha) + blue * alpha, 1, 1});
  }
  double alpha = mean_max ? std::min(d / mean_max, 1.0) : 1.0;
  return HSV2RGB({green * (1 - alpha) + red * alpha, 1, 1});
}

void MyViewer::fairMesh() {
  if (model_type != ModelType::MESH)
    return;

  emit startComputation(tr("Fairing mesh..."));
  OpenMesh::Smoother::JacobiLaplaceSmootherT<MyMesh> smoother(mesh);
  smoother.initialize(OpenMesh::Smoother::SmootherT<MyMesh>::Normal, // or: Tangential_and_Normal
                      OpenMesh::Smoother::SmootherT<MyMesh>::C1);
  for (size_t i = 1; i <= 10; ++i) {
    smoother.smooth(10);
    emit midComputation(i * 10);
  }
  updateMesh(false);
  emit endComputation();
}

void MyViewer::updateVertexNormals() {
  // Weights according to:
  //   N. Max, Weights for computing vertex normals from facet normals.
  //     Journal of Graphics Tools, Vol. 4(2), 1999.
  for (auto v : mesh.vertices()) {
    Vector n(0.0, 0.0, 0.0);
    for (auto h : mesh.vih_range(v)) {
      if (mesh.is_boundary(h))
        continue;
      auto in_vec  = mesh.calc_edge_vector(h);
      auto out_vec = mesh.calc_edge_vector(mesh.next_halfedge_handle(h));
      double w = in_vec.sqrnorm() * out_vec.sqrnorm();
      n += (in_vec % out_vec) / (w == 0.0 ? 1.0 : w);
    }
    double len = n.length();
    if (len != 0.0)
      n /= len;
    mesh.set_normal(v, n);
  }
}

void MyViewer::updateMesh(bool update_mean_range) {
  if (model_type == ModelType::BEZIER_SURFACE)
    generateBezierMesh();
  if (model_type == ModelType::TSPLINE_SURFACE)
	  generateTSplineMesh();
  mesh.request_face_normals(); mesh.request_vertex_normals();
  mesh.update_face_normals(); //mesh.update_vertex_normals();
  updateVertexNormals();
  updateMeanCurvature(update_mean_range);
}

void MyViewer::setupCamera() {
  // Set camera on the model
  Vector box_min, box_max;
  box_min = box_max = mesh.point(*mesh.vertices_begin());
  for (auto v : mesh.vertices()) {
    box_min.minimize(mesh.point(v));
    box_max.maximize(mesh.point(v));
  }
  camera()->setSceneBoundingBox(Vec(box_min.data()), Vec(box_max.data()));
  camera()->showEntireScene();

  slicing_scaling = 20 / (box_max - box_min).max();

  setSelectedName(-1);
  axes.shown = false;

  update();
}

bool MyViewer::openMesh(const std::string &filename) {
  if (!OpenMesh::IO::read_mesh(mesh, filename) || mesh.n_vertices() == 0)
    return false;
  model_type = ModelType::MESH;
  updateMesh();
  setupCamera();
  return true;
}

bool MyViewer::openBezier(const std::string &filename) {
  size_t n, m;
  try {
    std::ifstream f(filename.c_str());
    f.exceptions(std::ios::failbit | std::ios::badbit);
    f >> n >> m;
    degree[0] = n++; degree[1] = m++;
    bezier_control_points.resize(n * m);
    for (size_t i = 0, index = 0; i < n; ++i)
      for (size_t j = 0; j < m; ++j, ++index)
        f >> bezier_control_points[index][0] >> bezier_control_points[index][1] >> bezier_control_points[index][2];
  } catch(std::ifstream::failure &) {
    return false;
  }
  fileName = filename;
  model_type = ModelType::BEZIER_SURFACE;
  updateMesh();
  setupCamera();
  return true;
}

bool MyViewer::saveBezier(const std::string &filename) {
  if (model_type != ModelType::BEZIER_SURFACE)
    return false;

  try {
    std::ofstream f(filename.c_str());
    f.exceptions(std::ios::failbit | std::ios::badbit);
    f << degree[0] << ' ' << degree[1] << std::endl;
    for (const auto &p : bezier_control_points)
      f << p[0] << ' ' << p[1] << ' ' << p[2] << std::endl;
  } catch(std::ifstream::failure &) {
    return false;
  }
  return true;
}

bool MyViewer::openTSpline(const std::string &filename) {
	//https://www.geeksforgeeks.org/sparse-matrix-representations-set-3-csr/
	fileName = filename;

	size_t cpnum, ia_size;
	try {
		std::ifstream f(filename.c_str());
		f.exceptions(std::ios::failbit | std::ios::badbit);
		f >> cpnum >> ia_size;
		if (cpnum <= 0 || ia_size <= 0) return false;
		/*tspline_control_points.clear();
		IA.clear();
		JA.clear();
		si_array.clear();
		ti_array.clear();
		weights.clear();*/
		tspline_control_points.resize(cpnum);
		IA.resize(ia_size);
		JA.resize(cpnum);
		si_array.resize(cpnum);
		ti_array.resize(cpnum);
		weights.resize(cpnum);
		blend_functions.resize(cpnum);
		refined_points.resize(cpnum);
		refined_weights.resize(cpnum);
		for (size_t i = 0; i < cpnum; ++i){
			si_array[i].resize(5);
			ti_array[i].resize(5);
			f >> tspline_control_points[i][0] >> tspline_control_points[i][1] >> tspline_control_points[i][2];
			f >> si_array[i][0] >> si_array[i][1] >> si_array[i][2] >> si_array[i][3] >> si_array[i][4];
			f >> ti_array[i][0] >> ti_array[i][1] >> ti_array[i][2] >> ti_array[i][3] >> ti_array[i][4];
			f >> weights[i];
			//Filling up JA vector as well
			f >> JA[i];
			//Initializing blend function
			std::pair<std::vector<double>, std::vector<double>> blend_pair(si_array[i],ti_array[i]);
			blend_functions[i] = { blend_pair };
			//Initializing refined_points
			refined_points[i] = { tspline_control_points[i] };
			//Initializing refined_weights
			refined_weights[i] = { weights[i] };
		}
		//Finally filling up IA vector
		for (size_t i = 0; i < ia_size; i++)
			f >> IA[i];
	}
	catch (std::ifstream::failure &) {
		return false;
	}
	model_type = ModelType::TSPLINE_SURFACE;
	if (!checkTSplineCorrectness()) return false;
	updateEdgeTopology();
	if (!checkTSplineTopology()) return false;
	updateMesh();
	setupCamera();
	distMode = false;
	distColorMode = false;
	bringBackMode = false;
	return true;
}

bool MyViewer::saveTSpline(const std::string &filename) {
	if (model_type != ModelType::TSPLINE_SURFACE)
		return false;
	
	if (model_type == ModelType::TSPLINE_SURFACE) {
		try {
			std::ofstream f(filename.c_str());
			f.exceptions(std::ios::failbit | std::ios::badbit);
			size_t ia_size = IA.size();
			size_t cpnum = weights.size();
			f << cpnum << ' ' << ia_size << std::endl;
			for (size_t i = 0; i < cpnum; ++i) {
				f << tspline_control_points[i][0] << ' ' << tspline_control_points[i][1] << ' ' << tspline_control_points[i][2] << std::endl;
				f << si_array[i][0] << ' ' << si_array[i][1] << ' ' << si_array[i][2] << ' ' << si_array[i][3] << ' ' << si_array[i][4] << std::endl;
				f << ti_array[i][0] << ' ' << ti_array[i][1] << ' ' << ti_array[i][2] << ' ' << ti_array[i][3] << ' ' << ti_array[i][4] << std::endl;
				f << weights[i] << std::endl;
				//Save JA too
				f << JA[i] << std::endl;
			}
			//Finally, save IA
			for (size_t i = 0; i < ia_size; i++) {
				f << IA[i] << std::endl;
			}
		}
		catch (std::ifstream::failure &) {
			return false;
		}
		return true;
	}
}

void MyViewer::init() {
  glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, 1);

  QImage img(":/isophotes.png");
  glGenTextures(1, &isophote_texture);
  glBindTexture(GL_TEXTURE_2D, isophote_texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, img.width(), img.height(), 0, GL_BGRA,
               GL_UNSIGNED_BYTE, img.convertToFormat(QImage::Format_ARGB32).bits());

  glGenTextures(1, &slicing_texture);
  glBindTexture(GL_TEXTURE_1D, slicing_texture);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  static const unsigned char slicing_img[] = { 0b11111111, 0b00011100 };
  glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB, 2, 0, GL_RGB, GL_UNSIGNED_BYTE_3_3_2, &slicing_img);
}

void MyViewer::updateEdgeTopology() {
	edges.clear();
	int cpnum = tspline_control_points.size();
	int ia_size = IA.size();
	//Storing controlnet in row direction
	for (int i = 1; i < ia_size; ++i) {
		bool first = true;
		for (int j = IA[i - 1]; j < IA[i]; ++j) {
			//If last in row or is not connected with next in row /this way the topology is surely true to rule 2/
			if ((j == IA[i]-1) || (si_array[j][3] != si_array[j + 1][2])) {
				if (first) break;
				first = true;
				std::pair<int, int> ind_pair = std::pair<int, int>(j - 1, j);
				edges.push_back(ind_pair);
			}
			else {
				if (first) {
					first = false;
				}
				else {
					std::pair<int, int> ind_pair = std::pair<int, int>(j - 1, j);
					edges.push_back(ind_pair);
				}
			}
		}
	}

	//Drawing in column direction
	int col_num = *std::max_element(JA.begin(), JA.end()) + 1;
	bool first = true;
	for (int i = 0; i < col_num; ++i) {
		first = true;
		std::vector<int> col_indices = indicesOfColumn(i);
		for (int j = 0; j < col_indices.size(); ++j) {
			//If last or first in column or is not connected with next in column /this way the topology is surely true to rule 2/
			if (j==col_indices.size()-1 || j==0 || (ti_array[col_indices[j]][3] != ti_array[col_indices[j+1]][2])) {
				if (!first) {
					first = true;
					std::pair<int, int> ind_pair = std::pair<int, int>(col_indices[j - 1], col_indices[j]);
					edges.push_back(ind_pair);
				}
				else { if(j != (col_indices.size()-1) && ti_array[col_indices[j]][3] == ti_array[col_indices[j + 1]][2]) first = false; }
			}
			else {
				if (first) {
					first = false;
				}
				else {
					std::pair<int, int> ind_pair = std::pair<int, int>(col_indices[j - 1], col_indices[j]);
					edges.push_back(ind_pair);
				}
			}
		}
	}
}

//remove=true if we want to undo the temporary update which we did by remove=false
void MyViewer::updateEdgesTemporarily(bool remove, int temp_index) {
	for (int e = 0; e < edges.size(); e++) {
		if (edges[e].first >= temp_index) remove ? edges[e].first-- : edges[e].first++;
		if (edges[e].second >= temp_index) remove ? edges[e].second-- : edges[e].second++;
	}
}

void MyViewer::draw() {
  if (model_type == ModelType::BEZIER_SURFACE && show_control_points)
    drawBezierControlNet();
  if (model_type == ModelType::TSPLINE_SURFACE && show_control_points)
	  drawTSplineControlNet(false,-1);

  glPolygonMode(GL_FRONT_AND_BACK, !show_solid && show_wireframe ? GL_LINE : GL_FILL);
  glEnable(GL_POLYGON_OFFSET_FILL);
  glPolygonOffset(1, 1);

  if (show_solid || show_wireframe) {
    if (visualization == Visualization::PLAIN)
      glColor3d(1.0, 1.0, 1.0);
    else if (visualization == Visualization::ISOPHOTES) {
      glBindTexture(GL_TEXTURE_2D, isophote_texture);
      glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
      glEnable(GL_TEXTURE_2D);
      glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_SPHERE_MAP);
      glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_SPHERE_MAP);
      glEnable(GL_TEXTURE_GEN_S);
      glEnable(GL_TEXTURE_GEN_T);
    } else if (visualization == Visualization::SLICING) {
      glBindTexture(GL_TEXTURE_1D, slicing_texture);
      glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
      glEnable(GL_TEXTURE_1D);
    }
    for (auto f : mesh.faces()) {
      glBegin(GL_POLYGON);
      for (auto v : mesh.fv_range(f)) {
        if (visualization == Visualization::MEAN)
          glColor3dv(meanMapColor(mesh.data(v).mean));
        else if (visualization == Visualization::SLICING)
          glTexCoord1d(mesh.point(v) | slicing_dir * slicing_scaling);
        glNormal3dv(mesh.normal(v).data());
        glVertex3dv(mesh.point(v).data());
      }
      glEnd();
    }
    if (visualization == Visualization::ISOPHOTES) {
      glDisable(GL_TEXTURE_GEN_S);
      glDisable(GL_TEXTURE_GEN_T);
      glDisable(GL_TEXTURE_2D);
      glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    } else if (visualization == Visualization::SLICING) {
      glDisable(GL_TEXTURE_1D);
    }
  }

  if (show_solid && show_wireframe) {
    glPolygonMode(GL_FRONT, GL_LINE);
    glColor3d(0.0, 0.0, 0.0);
    glDisable(GL_LIGHTING);
    for (auto f : mesh.faces()) {
      glBegin(GL_POLYGON);
      for (auto v : mesh.fv_range(f))
        glVertex3dv(mesh.point(v).data());
      glEnd();
    }
    glEnable(GL_LIGHTING);
  }

  if (axes.shown)
    drawAxes();
}

void MyViewer::drawBezierControlNet() const {
  glDisable(GL_LIGHTING);
  glLineWidth(3.0);
  glColor3d(0.3, 0.3, 1.0);
  size_t m = degree[1] + 1;
  for (size_t k = 0; k < 2; ++k)
    for (size_t i = 0; i <= degree[k]; ++i) {
      glBegin(GL_LINE_STRIP);
      for (size_t j = 0; j <= degree[1-k]; ++j) {
        size_t const index = k ? j * m + i : i * m + j;
        const auto &p = bezier_control_points[index];
        glVertex3dv(p);
      }
      glEnd();
    }
  glLineWidth(1.0);
  glPointSize(8.0);
  glColor3d(1.0, 0.0, 1.0);
  glBegin(GL_POINTS);
  for (const auto &p : bezier_control_points)
    glVertex3dv(p);
  glEnd();
  glPointSize(1.0);
  glEnable(GL_LIGHTING);
}

void MyViewer::drawTSplineControlNet(bool with_names, int names_index) const{
	glDisable(GL_LIGHTING);
	glLineWidth(3.0);
	glColor3d(0.3, 0.3, 1.0);
	for (auto& ind_pair : edges) {
		if(with_names) glPushName(names_index++);
		glBegin(GL_LINES);
		const auto &p1 = tspline_control_points[ind_pair.first] / weights[ind_pair.first];
		const auto &p2 = tspline_control_points[ind_pair.second] / weights[ind_pair.second];
		glVertex3dv(p1);
		glVertex3dv(p2);
		glEnd();
		if(with_names) glPopName();
	}

	//Drawing points
	glLineWidth(1.0);
	glPointSize(8.0);
	glBegin(GL_POINTS);
	for (int i = 0; i < tspline_control_points.size(); i++) {
		if (distColorMode) {
			Vec pColor = distColors[i];
			glColor3d(pColor[0],pColor[1], pColor[2]);
		}
		else glColor3d(1.0, 0.0, 1.0);
		Vec coords = { (tspline_control_points[i] / weights[i]) };
		glVertex3dv(coords);
	}
	glEnd();
	glPointSize(1.0);
	glEnable(GL_LIGHTING);
}

void MyViewer::drawAxes() const {
  const Vec &p = axes.position;
  glColor3d(1.0, 0.0, 0.0);
  drawArrow(p, p + Vec(axes.size, 0.0, 0.0), axes.size / 50.0);
  glColor3d(0.0, 1.0, 0.0);
  drawArrow(p, p + Vec(0.0, axes.size, 0.0), axes.size / 50.0);
  glColor3d(0.0, 0.0, 1.0);
  drawArrow(p, p + Vec(0.0, 0.0, axes.size), axes.size / 50.0);
  glEnd();
}

void MyViewer::drawWithNames() {
  if (axes.shown)
    return drawAxesWithNames();

  switch (model_type) {
  case ModelType::NONE: break;
  case ModelType::MESH:
    if (!show_wireframe)
      return;
    for (auto v : mesh.vertices()) {
      glPushName(v.idx());
      glRasterPos3dv(mesh.point(v).data());
      glPopName();
    }
    break;
  case ModelType::BEZIER_SURFACE:
    if (!show_control_points)
      return;
    for (size_t i = 0, ie = bezier_control_points.size(); i < ie; ++i) {
      Vec const &p = bezier_control_points[i];
      glPushName(i);
      glRasterPos3fv(p);
      glPopName();
    }
    break;
  case ModelType::TSPLINE_SURFACE:
	  if (!show_control_points)
		  return;
	  for (size_t i = 0, ie = tspline_control_points.size(); i < ie; ++i) {
		  Vec const &p = tspline_control_points[i]/weights[i];
		  glPushName(i);
		  glRasterPos3fv(p);
		  glPopName();
	  }
	  drawTSplineControlNet(true, tspline_control_points.size());
	  break;
  }
}

void MyViewer::drawAxesWithNames() const {
  const Vec &p = axes.position;
  glPushName(0);
  drawArrow(p, p + Vec(axes.size, 0.0, 0.0), axes.size / 50.0);
  glPopName();
  glPushName(1);
  drawArrow(p, p + Vec(0.0, axes.size, 0.0), axes.size / 50.0);
  glPopName();
  glPushName(2);
  drawArrow(p, p + Vec(0.0, 0.0, axes.size), axes.size / 50.0);
  glPopName();
}

void MyViewer::endSelection(const QPoint &p) {
	
	glFlush();
	// Get the number of objects that were seen through the pick matrix frustum.
	// Resets GL_RENDER mode.
	GLint nbHits = glRenderMode(GL_RENDER);
	if (nbHits <= 0)
		setSelectedName(-1);
	else
	{
		if (model_type == ModelType::TSPLINE_SURFACE) {
			auto selBuff = selectBuffer();
			for (int i = 0; i < nbHits; ++i)
				//If a point is selected, too
				if (selBuff[i * 4 + 3] < tspline_control_points.size()) {
					setSelectedName(selBuff[i * 4 + 3]);
					return;
				}
			setSelectedName(selBuff[3]);
		}
	}
}

std::vector<int> MyViewer::indicesOfColumn(int colindex, bool inOrig) {
	std::vector<int> ret_vec;
	ret_vec.clear();
	for (int i = 0; i < (inOrig ? JAOrig.size() : JA.size()); i++) {
		if ((inOrig ? JAOrig[i] : JA[i]) == colindex) ret_vec.push_back(i);
	}
	return ret_vec;
}

//Returns with index of row
int MyViewer::getRowOfExisting(int index, bool inOrig) {
	int i = 0;
	for (; (inOrig ? IAOrig[i] : IA[i]) <= index; i++) {
	}
	return i - 1;
}

//Returns with true,i if row exists, false otherwise
//maxFromEquals false if we take the first equal as return row, true if we take the last
std::pair<bool, int> MyViewer::getRowOfNew(int first_row, int sec_row, double t, bool maxFromEquals) {
	//If inserting horizontally
	if (first_row == sec_row) return std::pair<bool, int>(true, first_row);
	//If inserting vertically
	//If first_row and sec_row next to each other, inserting between them
	if(first_row + 1 == sec_row) return std::pair<bool, int>(false, sec_row);
	else {
		int i = first_row+1;
		for (; sec_row >= i; i++) {
			if (!maxFromEquals && ti_array[IA[i]][2] == t) return std::pair<bool, int>(true, i == sec_row ? i-1 : i);
			if (ti_array[IA[i]][2] > t) {
				bool existing_row = ti_array[IA[i - 1]][2] == t;
				return std::pair<bool, int>(existing_row, existing_row ? i-1 : i);
			}
			//If t is same as t of sec_row
			else if (sec_row == i) {
				return std::pair<bool, int>(true, i - 1);
			}
		}
	}
}

//Returns with true,i if col exists, false otherwise
//maxFromEquals false if we take the first equal as return col, true if we take the last
std::pair<bool, int> MyViewer::getColOfNew(int first_col, int sec_col, double s, bool maxFromEquals) {
	//If inserting vertically
	if (first_col == sec_col) return std::pair<bool, int>(true,first_col);

	//Inserting horizontally
	//If new_col between the two endpoints
	if(first_col + 1 == sec_col) return std::pair<bool, int>(false, sec_col);
	else {
		int i = first_col+1;
		double last_s = 0.0;
		for (; sec_col >= i; i++) {
			auto col_ind = indicesOfColumn(i);
			if (!maxFromEquals && si_array[col_ind[0]][2] == s) return std::pair<bool, int>(true, i == sec_col ? i-1 : i);
			if (si_array[col_ind[0]][2] > s) {
				bool existing_col = last_s == s;
				return std::pair<bool, int>(existing_col, existing_col ? i - 1 : i);
			}
			//If s is same as s of sec_col
			else if (sec_col == i) {
				return std::pair<bool, int>(true, i-1);
			}
			last_s = si_array[col_ind[0]][2];
		}
	}

	
}

//Gives back index of new point based on the edge of the vertical insertion
//New insertion can only happen on edges with positive knot value
int MyViewer::getIndex(int first_row, int sec_row, int act_col, double t, bool maxFromEquals) {
	//Insertion on vertical edge? --i dont think it's necessary it's much easier to calculate on point

	//If inserting between two rows
	if (first_row + 1 == sec_row) return IA[sec_row];

	for (int i = first_row+1; i <= sec_row; ++i) {
		//Checking if in the same row as the one to be inserted
		//TODO check more securely
		if (!maxFromEquals && ti_array[IA[i]][2] == t) {
			//Case of zero interval insertion
			if (i == sec_row) return IA[sec_row];
			int j = IA[i];
			for (; j < IA[i + 1]; ++j) {
				if (JA[j] > act_col) {
					return j;
				}
			}
			return j;
		}
		else if (ti_array[IA[i]][2] > t || i == sec_row) {
			//New row
			if (ti_array[IA[i-1]][2] != t) return IA[i];
			int j = IA[i-1];
			for (; j < IA[i]; ++j) {
				if (JA[j] > act_col) {
					return j;
				}
			}
			return j;
		}
	}
}

void MyViewer::updateIA(int first_row, int sec_row, double t, bool maxFromEquals) {
	auto row = getRowOfNew(first_row, sec_row, t,maxFromEquals);
	//If is inserted into existing row
	if (row.first) {
		for (int j = row.second + 1; j < IA.size(); j++) {
			IA[j] += 1;
		}
	}
	else {
		for (int j = row.second; j < IA.size(); j++) {
			IA[j] += 1;
		}
		IA.insert(IA.begin() + row.second, IA[row.second] - 1);
	}
}

void MyViewer::deleteFromIA(int del_ind) {
	int row = getRowOfExisting(del_ind);
	//If it was the only one in the row
	if (IA[row] == del_ind && IA[row + 1] == del_ind + 1) {
		for (int i = row + 1; i < IA.size(); ++i) {
			IA[i]--;
		}
		IA.erase(IA.begin() + row);
	}
	else{
		int i = row + 1;
		for (; i < IA.size(); ++i) {
			IA[i]--;
		}
	}
}

void MyViewer::updateJA(int first_col, int sec_col, int new_ind, double s, bool maxFromEquals) {
	auto col = getColOfNew(first_col, sec_col,s, maxFromEquals);
	//If is inserted into existing col
	if (col.first) {
		JA.insert(JA.begin() + new_ind, col.second);
	}
	else {
		for (int j = 0; j < JA.size(); j++) {
			if (JA[j] >= col.second) JA[j] += 1;
		}
		JA.insert(JA.begin() + new_ind, col.second);
	}
}

void MyViewer::deleteFromJA(int del_ind) {
	int col = JA[del_ind];
	JA.erase(JA.begin() + del_ind);
	//If it was the only one in the col
	if (std::find(JA.begin(), JA.end(), col) == JA.end()) {
		for (int i = 0; i < JA.size(); ++i) {
			if (JA[i] > col) {
				JA[i]--;
			}
		}
	}
}

bool MyViewer::edgeExists(int first_ind, int sec_ind) {
	std::pair<int, int> temp_pair = std::pair<int, int>(first_ind, sec_ind);
	return (std::find(edges.begin(), edges.end(), temp_pair) != edges.end());
}

//returns true if violated and the newlyAdded vec refreshed
std::pair<bool, std::pair<std::vector<int>, std::vector<int>>> MyViewer::checkForViol1(std::vector<int> excluded, std::vector<int> newlyAdded) {
	bool violated = false;
	int origExcSize = excluded.size();
	int cpnum = tspline_control_points.size();
	for (int i = 0; i < cpnum;i++) {
		//Check only for points which are not in the excluded
		if (std::find(excluded.begin(), excluded.end(), i) == excluded.end()) {
			int act_row = getRowOfExisting(i);
			int act_col = JA[i];
			int jmax = blend_functions[i].size();
			for (int j = 0; j < jmax; j++) {
				auto bf = blend_functions[i][j];
				bool actualAddedToOther = false;
				auto ts_down = checkTsDown(act_row, act_col, i, bf.first, bf.second, 1, newlyAdded);
				if (!ts_down.first.first) {
					//Refine blend func of point i by inserting at ts_down.second.first + 1 value ts_down.second.second
					violated = true;

					//Indices downwards
					auto col_inds = indicesOfColumn(JA[i]);
					auto act_vec = std::find(col_inds.begin(), col_inds.end(), i);
					//If no point below OR
					//If there is no point in the first t down but there is on the second down which causes a violation
					if (act_vec == col_inds.begin() || getRowOfExisting(*(act_vec - 1)) != ts_down.first.second.first) {
						//force inserting a point first place
						//Two cases: no point and first t violated or no point but only the second t violated
						double new_value = ts_down.second.first == 1 ? ts_down.second.second : bf.second[1];
						int new_index = getIndex(ts_down.first.second.first-1,act_row,act_col,new_value, true);
						updateJA(act_col, act_col, new_index, bf.first[2], false);
						updateIA(ts_down.first.second.first - 1, act_row, new_value, true);
						std::vector<double> new_ti = { bf.second[0], bf.second[0], new_value, bf.second[2], bf.second[3]};

						auto ret_pair = insertAfterViol(new_index, bf.first, new_ti, excluded, newlyAdded);
						excluded = ret_pair.first;
						newlyAdded = ret_pair.second;
						//Handle index changes due to new point inserted- to still check all points
						cpnum++;
						if (new_index <= i) i++;
						break;
					}

					auto refined_pairs = refineBlend(bf.second, ts_down.second.first + 1, ts_down.second.second);
					//Two insertions:
					//First: refining the actual in t direction-> refine the blend function, +multipl*d to blendMultipliers[i]
					//Delete actual old blend func
					blend_functions[i].erase(blend_functions[i].begin() + j);
					Vec temp_point = refined_points[i][j];
					double temp_weight = refined_weights[i][j];
					refined_points[i].erase(refined_points[i].begin() + j);
					refined_weights[i].erase(refined_weights[i].begin() + j);
					//Finding the blend function which is the same, if doesn't exist, add new one
					bool exists = false;
					for (int k = 0; k < blend_functions[i].size(); k++) {
						auto temp_bf = blend_functions[i][k];
						//TODO does this check equality in the right way??
						if (temp_bf.first == bf.first && temp_bf.second == refined_pairs.second.second) {
							refined_points[i][k] += temp_point * refined_pairs.second.first;
							refined_weights[i][k] += temp_weight * refined_pairs.second.first;
							exists = true;
							jmax--;
							actualAddedToOther = true;
							break;
						}
					}
					if (!exists) {
						std::pair<std::vector<double>, std::vector<double>> blend_pair(bf.first, refined_pairs.second.second);
						blend_functions[i].insert(blend_functions[i].begin() + j,blend_pair);
						refined_points[i].insert(refined_points[i].begin() + j, temp_point * refined_pairs.second.first);
						refined_weights[i].insert(refined_weights[i].begin() + j, temp_weight * refined_pairs.second.first);
						bf = blend_functions[i][j];
					}
					//Update M matrix
					if (bringBackMode) updateM(indsInOrig[i], indsInOrig[i], refined_pairs.second.first);
					//Second : getIndex of(bf.first[2], ts_down.second.second) if inserting below the middle point
					int ref_ind = *(act_vec - 1);

					//Update M matrix
					if (bringBackMode) updateM(indsInOrig[i], indsInOrig[ref_ind], refined_pairs.first.first);

					//refine the blend function
					//with the proper index see if any of the existing blends is the same as new one ->+ multipl*c to blend_multipliers[ref_ind][ind of same blend]
					//else store new one too

					//If pushing to point which was newly added(is in excluded) and is first one to push to it(size of blend_func[ref_ind] is 1)
					//Then delete old and replace with new
					if (std::find(excluded.begin(), excluded.end(), ref_ind) != excluded.end() && blend_functions[ref_ind].size() == 1 && refined_weights[ref_ind][0] == 0) {
						blend_functions[ref_ind].erase(blend_functions[ref_ind].begin());
						refined_points[ref_ind].erase(refined_points[ref_ind].begin());
						refined_weights[ref_ind].erase(refined_weights[ref_ind].begin());
						std::pair<std::vector<double>, std::vector<double>> blend_pair(bf.first, refined_pairs.first.second);
						blend_functions[ref_ind].push_back(blend_pair);
						refined_points[ref_ind].push_back(temp_point *refined_pairs.first.first);
						refined_weights[ref_ind].push_back(temp_weight *refined_pairs.first.first);
					}
					else {
						exists = false;
						for (int k = 0; k < blend_functions[ref_ind].size(); k++) {
							auto temp_bf = blend_functions[ref_ind][k];
							//TODO does this check equality in the right way??
							if (temp_bf.first == bf.first && temp_bf.second == refined_pairs.first.second) {
								refined_points[ref_ind][k] += temp_point * refined_pairs.first.first;
								refined_weights[ref_ind][k] += temp_weight * refined_pairs.first.first;
								exists = true;
								break;
							}
						}
						if (!exists) {
							std::pair<std::vector<double>, std::vector<double>> blend_pair(bf.first, refined_pairs.first.second);
							blend_functions[ref_ind].push_back(blend_pair);
							refined_points[ref_ind].push_back(temp_point *refined_pairs.first.first);
							refined_weights[ref_ind].push_back(temp_weight *refined_pairs.first.first);
						}
					}
				}
				
				if (actualAddedToOther) {
					j--;
					break;
				}
				else {
					bf = blend_functions[i][j];
				}
				auto ts_up = checkTsUp(act_row, act_col, i, bf.first, bf.second, 1, newlyAdded);
				if (!ts_up.first.first) {
					//Refine blend func of point i by inserting at ts_up.second.first value ts_up.second.second
					violated = true;

					//Indices downwards
					auto col_inds = indicesOfColumn(JA[i]);
					auto act_vec = std::find(col_inds.begin(), col_inds.end(), i);
					//If no point after it in the col OR
					//If there is no point in the first t down but there is on the second down which causes a violation
					if (act_vec == col_inds.end() - 1 || getRowOfExisting(*(act_vec + 1)) != ts_up.first.second.first) {
						//force inserting a point first place
						//Two cases: no point and first t violated or no point but only the second t violated
						double new_value = ts_up.second.first == 3 ? ts_up.second.second : bf.second[3];
						int new_index = getIndex(act_row, ts_up.first.second.first + 1, act_col, new_value, false);
						updateJA(act_col, act_col, new_index, bf.first[2], false);
						updateIA(act_row, ts_up.first.second.first + 1, new_value, false);
						std::vector<double> new_ti = { bf.second[1], bf.second[2], new_value, bf.second[4], bf.second[4] };

						auto ret_pair = insertAfterViol(new_index, bf.first, new_ti, excluded, newlyAdded);
						excluded = ret_pair.first;
						newlyAdded = ret_pair.second;
						//Handle index changes due to new point inserted- to still check all points
						cpnum++;
						if (new_index <= i) i++;
						break;
					}

					auto refined_pairs = refineBlend(bf.second, ts_up.second.first, ts_up.second.second);
					//Two insertions:
					//First: refining the actual in t direction-> refine the blend function, +multipl*c to blendMultipliers[i]
					//Delete actual old blend func
					blend_functions[i].erase(blend_functions[i].begin() + j);
					Vec temp_point = refined_points[i][j];
					double temp_weight = refined_weights[i][j];
					refined_points[i].erase(refined_points[i].begin() + j);
					refined_weights[i].erase(refined_weights[i].begin() + j);
					//Finding the blend function which is the same, if doesn't exist, add new one
					bool exists = false;
					for (int k = 0; k < blend_functions[i].size(); k++) {
						auto temp_bf = blend_functions[i][k];
						//TODO does this check equality in the right way??
						if (temp_bf.first == bf.first && temp_bf.second == refined_pairs.first.second) {
							refined_points[i][k] += temp_point * refined_pairs.first.first;
							refined_weights[i][k] += temp_weight * refined_pairs.first.first;
							exists = true;
							jmax--;
							actualAddedToOther = true;
							break;
						}
					}
					if (!exists) {
						std::pair<std::vector<double>, std::vector<double>> blend_pair(bf.first, refined_pairs.first.second);
						blend_functions[i].insert(blend_functions[i].begin() + j, blend_pair);
						refined_points[i].insert(refined_points[i].begin() + j, temp_point *refined_pairs.first.first);
						refined_weights[i].insert(refined_weights[i].begin() + j, temp_weight * refined_pairs.first.first);
						bf = blend_functions[i][j];
					}
					//Update M matrix
					if (bringBackMode) updateM(indsInOrig[i], indsInOrig[i], refined_pairs.first.first);

					//Second : getIndex of(bf.first[2], ts_up.second.second) if inserting above the middle point
					int ref_ind = *(act_vec + 1);

					//Update M matrix
					if (bringBackMode) updateM(indsInOrig[i], indsInOrig[ref_ind], refined_pairs.second.first);

					//refine the blend function
					//with the proper index see if any of the existing blends is the same as new one ->+ d to blend_multipliers[ref_ind][ind of same blend]
					//else store new one too

					//If pushing to point which was newly added(is in excluded) and is first one to push to it(size of blend_func[ref_ind] is 1)
					//Then delete old and replace with new
					if (std::find(excluded.begin(), excluded.end(), ref_ind) != excluded.end() && blend_functions[ref_ind].size() == 1 && refined_weights[ref_ind][0] == 0) {
						blend_functions[ref_ind].erase(blend_functions[ref_ind].begin());
						refined_points[ref_ind].erase(refined_points[ref_ind].begin());
						refined_weights[ref_ind].erase(refined_weights[ref_ind].begin());
						std::pair<std::vector<double>, std::vector<double>> blend_pair(bf.first, refined_pairs.second.second);
						blend_functions[ref_ind].push_back(blend_pair);
						refined_points[ref_ind].push_back(temp_point *refined_pairs.second.first);
						refined_weights[ref_ind].push_back(temp_weight *refined_pairs.second.first);
					}
					else {
						bool exists = false;
						for (int k = 0; k < blend_functions[ref_ind].size(); k++) {
							auto temp_bf = blend_functions[ref_ind][k];
							//TODO does this check equality in the right way??
							if (temp_bf.first == bf.first && temp_bf.second == refined_pairs.second.second) {
								refined_points[ref_ind][k] += temp_point * refined_pairs.second.first;
								refined_weights[ref_ind][k] += temp_weight * refined_pairs.second.first;
								exists = true;
								break;
							}
						}
						if (!exists) {
							std::pair<std::vector<double>, std::vector<double>> blend_pair(bf.first, refined_pairs.second.second);
							blend_functions[ref_ind].push_back(blend_pair);
							refined_points[ref_ind].push_back(temp_point *refined_pairs.second.first);
							refined_weights[ref_ind].push_back(temp_weight *refined_pairs.second.first);
						}
					}
				}
				
				if (actualAddedToOther) {
					j--;
					break;
				}
				else {
					bf = blend_functions[i][j];
				}
				auto ss_down = checkSsDown(act_row, act_col, i, bf.first, bf.second, 1, newlyAdded);
				if (!ss_down.first.first) {
					//Refine blend func of point i by inserting at ss_down.second.first + 1 value ss_down.second.second
					violated = true;

					//If no point before it OR
					//If there is no point in the first s down but there is on the second down which causes a violation
					if (i == IA[act_row] || JA[i - 1] != ss_down.first.second.first) {
						//force inserting a point first place
						//Two cases: no point and first s violated or no point but only the second s violated
						double new_value = ss_down.second.first == 1 ? ss_down.second.second : bf.first[1];
						int new_index = i;
						updateJA(ss_down.first.second.first - 1, act_col, new_index, new_value, true);
						updateIA(act_row, act_row, bf.second[2], true);
						std::vector<double> new_si = { bf.first[0], bf.first[0], new_value, bf.first[2], bf.first[3] };

						auto ret_pair = insertAfterViol(new_index, new_si, bf.second, excluded, newlyAdded);
						excluded = ret_pair.first;
						newlyAdded = ret_pair.second;
						//Handle index changes due to new point inserted- to still check all points
						cpnum++;
						if (new_index <= i) i++;
						break;
					}

					auto refined_pairs = refineBlend(bf.first, ss_down.second.first + 1, ss_down.second.second);
					//Two insertions:
					//First: refining the actual in s direction-> refine the blend function, +multipl*d to blendMultipliers[i]
					//Delete actual old blend func
					blend_functions[i].erase(blend_functions[i].begin() + j);
					Vec temp_point = refined_points[i][j];
					double temp_weight = refined_weights[i][j];
					refined_points[i].erase(refined_points[i].begin() + j);
					refined_weights[i].erase(refined_weights[i].begin() + j);
					//Finding the blend function which is the same, if doesn't exist, add new one
					bool exists = false;
					for (int k = 0; k < blend_functions[i].size(); k++) {
						auto temp_bf = blend_functions[i][k];
						//TODO does this check equality in the right way??
						if (temp_bf.first == refined_pairs.second.second && temp_bf.second == bf.second) {
							refined_points[i][k] += temp_point * refined_pairs.second.first;
							refined_weights[i][k] += temp_weight * refined_pairs.second.first;
							exists = true;
							jmax--;
							actualAddedToOther = true;
							break;
						}
					}
					if (!exists) {
						std::pair<std::vector<double>, std::vector<double>> blend_pair(refined_pairs.second.second, bf.second);
						blend_functions[i].insert(blend_functions[i].begin() + j, blend_pair);
						refined_points[i].insert(refined_points[i].begin() + j, temp_point *refined_pairs.second.first);
						refined_weights[i].insert(refined_weights[i].begin() + j, temp_weight * refined_pairs.second.first);
						bf = blend_functions[i][j];
					}
					//Update M matrix
					if (bringBackMode) updateM(indsInOrig[i], indsInOrig[i], refined_pairs.second.first);

					//Second : getIndex of(ss_down.second.second,bf.second[2]) if inserting below the middle point
					int ref_ind = i-1;

					//Update M matrix
					if (bringBackMode) updateM(indsInOrig[i], indsInOrig[ref_ind], refined_pairs.first.first);

					//refine the blend function
					//with the proper index see if any of the existing blends is the same as new one ->+ c to blend_multipliers[ref_ind][ind of same blend]
					//else store new one too

					//If pushing to point which was newly added(is in excluded) and is first one to push to it(size of blend_func[ref_ind] is 1)
					//Then delete old and replace with new
					if (std::find(excluded.begin(), excluded.end(), ref_ind) != excluded.end() && blend_functions[ref_ind].size() == 1 && refined_weights[ref_ind][0] == 0) {
						blend_functions[ref_ind].erase(blend_functions[ref_ind].begin());
						refined_points[ref_ind].erase(refined_points[ref_ind].begin());
						refined_weights[ref_ind].erase(refined_weights[ref_ind].begin());
						std::pair<std::vector<double>, std::vector<double>> blend_pair(refined_pairs.first.second, bf.second);
						blend_functions[ref_ind].push_back(blend_pair);
						refined_points[ref_ind].push_back(temp_point *refined_pairs.first.first);
						refined_weights[ref_ind].push_back(temp_weight *refined_pairs.first.first);
					}
					else {
						bool exists = false;
						for (int k = 0; k < blend_functions[ref_ind].size(); k++) {
							auto temp_bf = blend_functions[ref_ind][k];
							//TODO does this check equality in the right way??
							if (temp_bf.first == refined_pairs.first.second && temp_bf.second == bf.second) {
								refined_points[ref_ind][k] += temp_point * refined_pairs.first.first;
								refined_weights[ref_ind][k] += temp_weight * refined_pairs.first.first;
								exists = true;
								break;
							}
						}
						if (!exists) {
							std::pair<std::vector<double>, std::vector<double>> blend_pair(refined_pairs.first.second, bf.second);
							blend_functions[ref_ind].push_back(blend_pair);
							refined_points[ref_ind].push_back(temp_point *refined_pairs.first.first);
							refined_weights[ref_ind].push_back(temp_weight *refined_pairs.first.first);
						}
					}
				}
				
				if (actualAddedToOther) {
					j--;
					break;
				}
				else {
					bf = blend_functions[i][j];
				}
				auto ss_up = checkSsUp(act_row, act_col, i, bf.first, bf.second, 1, newlyAdded);
				if (!ss_up.first.first) {
					//Refine blend func of point i by inserting at ss_up.second.first value ss_up.second.second
					violated = true;

					//If no point after it int the row OR
					//If there is no point in the first s up but there is on the second up which causes a violation
					if (i + 1 == IA[act_row + 1] || JA[i + 1] != ss_up.first.second.first) {
						//force inserting a point first place
						int new_index = i+1;
						//Two cases: no point and first s violated or no point but only the second s violated
						double new_value = ss_up.second.first == 3 ? ss_up.second.second : bf.first[3];
						updateJA(act_col, ss_up.first.second.first + 1, new_index, new_value, false);
						updateIA(act_row, act_row, bf.second[2], false);
						std::vector<double> new_si = { bf.first[1], bf.first[2], new_value, bf.first[4], bf.first[4] };

						auto ret_pair = insertAfterViol(new_index, new_si, bf.second, excluded, newlyAdded);
						excluded = ret_pair.first;
						newlyAdded = ret_pair.second;
						//Handle index changes due to new point inserted- to still check all points
						cpnum++;
						if (new_index <= i) i++;
						break;
					}

					auto refined_pairs = refineBlend(bf.first, ss_up.second.first, ss_up.second.second);
					//Two insertions:
					//First: refining the actual in s direction-> refine the blend function, +multipl*c to blendMultipliers[i]
					//Delete actual old blend func
					blend_functions[i].erase(blend_functions[i].begin() + j);
					Vec temp_point = refined_points[i][j];
					double temp_weight = refined_weights[i][j];
					refined_points[i].erase(refined_points[i].begin() + j);
					refined_weights[i].erase(refined_weights[i].begin() + j);
					//Finding the blend function which is the same, if doesn't exist, add new one
					bool exists = false;
					for (int k = 0; k < blend_functions[i].size(); k++) {
						auto temp_bf = blend_functions[i][k];
						//TODO does this check equality in the right way??
						if (temp_bf.first == refined_pairs.first.second && temp_bf.second == bf.second) {
							refined_points[i][k] += temp_point * refined_pairs.first.first;
							refined_weights[i][k] += temp_weight * refined_pairs.first.first;
							exists = true;
							jmax--;
							j--;
							break;
						}
					}
					if (!exists) {
						std::pair<std::vector<double>, std::vector<double>> blend_pair(refined_pairs.first.second, bf.second);
						blend_functions[i].insert(blend_functions[i].begin() + j, blend_pair);
						refined_points[i].insert(refined_points[i].begin() + j, temp_point *refined_pairs.first.first);
						refined_weights[i].insert(refined_weights[i].begin() + j, temp_weight * refined_pairs.first.first);
						bf = blend_functions[i][j];
					}
					//Update M matrix
					if (bringBackMode) updateM(indsInOrig[i], indsInOrig[i], refined_pairs.first.first);

					//Second : getIndex of(ss_up.second.second,bf.second[2]) if inserting above the middle point
					int ref_ind = i + 1;

					//Update M matrix
					if (bringBackMode) updateM(indsInOrig[i], indsInOrig[ref_ind], refined_pairs.second.first);

					//refine the blend function
					//with the proper index see if any of the existing blends is the same as new one ->+ d to blend_multipliers[ref_ind][ind of same blend]
					//else store new one too

					//If pushing to point which was newly added(is in excluded) and is first one to push to it(size of blend_func[ref_ind] is 1)
					//Then delete old and replace with new
					if (std::find(excluded.begin(), excluded.end(), ref_ind) != excluded.end() && blend_functions[ref_ind].size() == 1 && refined_weights[ref_ind][0] == 0) {
						blend_functions[ref_ind].erase(blend_functions[ref_ind].begin());
						refined_points[ref_ind].erase(refined_points[ref_ind].begin());
						refined_weights[ref_ind].erase(refined_weights[ref_ind].begin());
						std::pair<std::vector<double>, std::vector<double>> blend_pair(refined_pairs.second.second, bf.second);
						blend_functions[ref_ind].push_back(blend_pair);
						refined_points[ref_ind].push_back(temp_point *refined_pairs.second.first);
						refined_weights[ref_ind].push_back(temp_weight *refined_pairs.second.first);
					}
					else {
						bool exists = false;
						for (int k = 0; k < blend_functions[ref_ind].size(); k++) {
							auto temp_bf = blend_functions[ref_ind][k];
							//TODO does this check equality in the right way??
							if (temp_bf.first == refined_pairs.second.second && temp_bf.second == bf.second) {
								refined_points[ref_ind][k] += temp_point * refined_pairs.second.first;
								refined_weights[ref_ind][k] += temp_weight * refined_pairs.second.first;
								exists = true;
								break;
							}
						}
						if (!exists) {
							std::pair<std::vector<double>, std::vector<double>> blend_pair(refined_pairs.second.second, bf.second);
							blend_functions[ref_ind].push_back(blend_pair);
							refined_points[ref_ind].push_back(temp_point *refined_pairs.second.first);
							refined_weights[ref_ind].push_back(temp_weight *refined_pairs.second.first);
						}
					}
				}
			}
		}
	}

	excluded.erase(excluded.begin(), excluded.begin() + origExcSize);
	return std::pair<bool, std::pair<std::vector<int>, std::vector<int>>>(violated, std::pair<std::vector<int>, std::vector<int>>(excluded,newlyAdded));
}

std::pair<bool, std::pair<std::vector<int>, std::vector<int>>> MyViewer::checkForViol2(std::vector<int> excluded, std::vector<int> newlyAdded) {
	bool violated = false;
	int cpnum = tspline_control_points.size();
	for (int i = 0; i < cpnum; i++) {
		//Check only for points which are not in the excluded, excluded is going to contain the indices of the newly added 
		if (std::find(excluded.begin(), excluded.end(), i) == excluded.end()) {
			int act_row = getRowOfExisting(i);
			int act_col = JA[i];
			for (int j = 0; j < blend_functions[i].size(); j++) {
				auto bf = blend_functions[i][j];
				bool needsOneMoreIter = false;
				auto ts_down = checkTsDown(act_row, act_col, i, bf.first, bf.second, 2, newlyAdded);
				if (!ts_down.first.first) {
					//Insert new point at getIndex(bf.first[2],bf.second[ts_down.second.first])
					violated = true;
					needsOneMoreIter = true;
					int new_index;
					std::vector<double> new_ti;
					//if inserting with 1 below the middle point
					if (ts_down.second.first == 1) {
						new_index = getIndex(ts_down.first.second.first, act_row, act_col, bf.second[ts_down.second.first], true);
						updateJA(act_col, act_col, new_index, bf.first[2], true);
						updateIA(ts_down.first.first, act_row, bf.second[ts_down.second.first], true);
						new_ti = { ts_down.second.second, ts_down.second.second, bf.second[ts_down.second.first], bf.second[2], bf.second[3] };
					}
					//if inserting with 2 below the middle point
					else {
						auto col_inds = indicesOfColumn(JA[i]);
						auto act_vec = std::find(col_inds.begin(), col_inds.end(), i);
						//If the first hit has same t as the one to be inserted and there is at least one row between them, then insert the point before the first hit
						if (bf.second[0] == bf.second[1] && act_row-ts_down.first.second.first > 1) {
							new_index = getIndex(ts_down.first.second.first, act_row, act_col, bf.second[1], true);
							updateJA(act_col, act_col, new_index, bf.first[2], true);
							updateIA(ts_down.first.first, act_row, bf.second[1], true);
							new_ti = { ts_down.second.second, ts_down.second.second, bf.second[1], bf.second[2], bf.second[3] };
						}
						//If there is no point in the first t down which doesn't cause violation but the second down causes a violation
						//In this case we insert a point on the first t down
						else if (act_vec == col_inds.begin() || getRowOfExisting(*(act_vec - 1)) != ts_down.first.second.first) {
							new_index = getIndex(ts_down.first.second.second, act_row, act_col, bf.second[1], true);
							updateJA(act_col, act_col, new_index, bf.first[2], true);
							updateIA(ts_down.first.second.second, act_row, bf.second[1], true);
							new_ti = { ts_down.second.second, ts_down.second.second, bf.second[1], bf.second[2], bf.second[3] };
						}
						else {
							new_index = getIndex(ts_down.first.second.second, ts_down.first.second.first, act_col, bf.second[ts_down.second.first], true);
							updateJA(act_col, act_col, new_index, bf.first[2], true);
							updateIA(ts_down.first.second.second, ts_down.first.second.first, bf.second[ts_down.second.first], true);
							new_ti = { ts_down.second.second, ts_down.second.second, bf.second[ts_down.second.first], bf.second[1], bf.second[2] };
						}
					}
					
					auto ret_pair = insertAfterViol(new_index,bf.first,new_ti,excluded,newlyAdded);
					excluded = ret_pair.first;
					newlyAdded = ret_pair.second;

					//Handle index changes due to new point inserted- to still check all points
					cpnum++;
					if (new_index <= i) i++;
					
				}

				auto ts_up = checkTsUp(act_row, act_col, i, bf.first, bf.second, 2, newlyAdded);
				if (!ts_up.first.first) {
					//Insert new point at getIndex(bf.first[2],bf.second[ts_up.second.first])
					violated = true;
					needsOneMoreIter = true;
					int new_index;
					std::vector<double> new_ti;
					//if inserting with 1 above the middle point
					if (ts_up.second.first == 3) {
						new_index = getIndex(act_row, ts_up.first.second.first, act_col, bf.second[ts_up.second.first], false);
						updateJA(act_col, act_col, new_index, bf.first[2], false);
						updateIA(act_row, ts_up.first.second.first, bf.second[ts_up.second.first], false);
						new_ti = { bf.second[1], bf.second[2], bf.second[ts_up.second.first], ts_up.second.second, ts_up.second.second };
					}
					//if inserting with 2 above the middle point
					else {
						auto col_inds = indicesOfColumn(JA[i]);
						auto act_vec = std::find(col_inds.begin(), col_inds.end(), i);
						//If the first hit has same s as the one to be inserted and there is at least one row between them, then insert the point before the first hit
						if (bf.second[3] == bf.second[4] && ts_up.first.second.first - act_row > 1) {
							new_index = getIndex(act_row, ts_up.first.second.first, act_col, bf.second[3], false);
							updateJA(act_col, act_col, new_index, bf.first[2], false);
							updateIA(act_row, ts_up.first.second.first, bf.second[3], false);
							new_ti = { bf.second[1], bf.second[2], bf.second[3], ts_up.second.second, ts_up.second.second };
						}
						//If there is no point in the first t up which doesn't cause violation but the second up causes a violation
						//In this case we insert a point on first t up
						else if (act_vec == col_inds.end()-1 || getRowOfExisting(*(act_vec + 1)) != ts_up.first.second.first) {
							new_index = getIndex(act_row, ts_up.first.second.second, act_col, bf.second[3], false);
							updateJA(act_col, act_col, new_index, bf.first[2], false);
							updateIA(act_row, ts_up.first.second.second, bf.second[3], false);
							new_ti = { bf.second[1], bf.second[2], bf.second[3], ts_up.second.second, ts_up.second.second };
						}
						else {
							new_index = getIndex(ts_up.first.second.first, ts_up.first.second.second, act_col, bf.second[ts_up.second.first], false);
							updateJA(act_col, act_col, new_index, bf.first[2], false);
							updateIA(ts_up.first.second.first, ts_up.first.second.second, bf.second[ts_up.second.first], false);
							new_ti = { bf.second[2], bf.second[3], bf.second[ts_up.second.first], ts_up.second.second, ts_up.second.second };
						}
					}
					
					auto ret_pair = insertAfterViol(new_index, bf.first, new_ti, excluded, newlyAdded);
					excluded = ret_pair.first;
					newlyAdded = ret_pair.second;

					//Handle index changes due to new point inserted- to still check all points
					cpnum++;
					if (new_index <= i) i++;
				}

				auto ss_down = checkSsDown(act_row, act_col, i, bf.first, bf.second, 2, newlyAdded);
				if (!ss_down.first.first) {
					//Insert new point at getIndex(bf.first[ss_down.second.first],bf.second[2])
					violated = true;
					needsOneMoreIter = true;
					int new_index;
					std::vector<double> new_si;
					//if inserting with 1 below the middle point
					if (ss_down.second.first == 1) {
						new_index = i;
						updateJA(ss_down.first.second.first, act_col, new_index, bf.first[ss_down.second.first], true);
						updateIA(act_row, act_row, bf.second[2], true);
						new_si = { ss_down.second.second, ss_down.second.second, bf.first[ss_down.second.first], bf.first[2], bf.first[3] };
					}
					//if inserting with 2 below the middle point
					else {
						//If the first hit has same t as the one to be inserted and there is at least one col between them, then insert the point before the first hit
						if (bf.first[0] == bf.first[1] && act_col - ss_down.first.second.first > 1) {
							new_index = i;
							updateJA(ss_down.first.second.first, act_col, new_index, bf.first[1], true);
							updateIA(act_row, act_row, bf.second[2], true);
							new_si = { ss_down.second.second, ss_down.second.second, bf.first[1], bf.first[2], bf.first[3] };
						}
						//If there is no point in the first s down which doesn't cause violation but the second down causes a violation
						//In this case we insert a point on the first s down
						else if (i == IA[act_row] || JA[i - 1] != ss_down.first.second.first) {
							new_index = i;
							updateJA(ss_down.first.second.second, act_col, new_index, bf.first[1], true);
							updateIA(act_row, act_row, bf.second[2], true);
							new_si = { ss_down.second.second, ss_down.second.second, bf.first[1], bf.first[2], bf.first[3] };
						}
						else {
							//If no point on the first s down, then i else i-1
							new_index = (getRowOfExisting(i - 1) == act_row && si_array[i - 1][2] >= bf.first[ss_down.second.first]) ? i - 1 : i;
							updateJA(ss_down.first.second.second, ss_down.first.second.first, new_index, bf.first[ss_down.second.first], true);
							updateIA(act_row, act_row, bf.second[2], true);
							new_si = { ss_down.second.second, ss_down.second.second, bf.first[ss_down.second.first], bf.first[1], bf.first[2] };
						}
					}
					
					auto ret_pair = insertAfterViol(new_index, new_si, bf.second, excluded, newlyAdded);
					excluded = ret_pair.first;
					newlyAdded = ret_pair.second;

					//Handle index changes due to new point inserted- to still check all points
					cpnum++;
					if (new_index <= i) i++;
				}

				auto ss_up = checkSsUp(act_row, act_col, i, bf.first, bf.second, 2, newlyAdded);
				if (!ss_up.first.first) {
					//Insert new point at getIndex(bf.first[ss_up.second.first],bf.second[2])
					violated = true;
					needsOneMoreIter = true;
					//if inserting with 1 above the middle point
					int new_index;
					std::vector<double> new_si;
					if (ss_up.second.first == 3) {
						new_index = i + 1;
						updateJA(act_col, ss_up.first.second.first, new_index, bf.first[ss_up.second.first], false);
						updateIA(act_row, act_row, bf.second[2], false);
						new_si = { bf.first[1], bf.first[2], bf.first[ss_up.second.first], ss_up.second.second, ss_up.second.second };
					}
					//if inserting with 2 above the middle point
					else {
						//If the first hit has same t as the one to be inserted and there is at least one row between them, then insert the point before the first hit
						if (bf.first[3] == bf.first[4] && ss_down.first.second.first - act_col > 1) {
							new_index = i + 1;
							updateJA(act_col, ss_up.first.second.first, new_index, bf.first[3], false);
							updateIA(act_row, act_row, bf.second[2], false);
							new_si = { bf.first[1], bf.first[2], bf.first[3], ss_up.second.second, ss_up.second.second };
						}
						//If there is no point in the first s down which doesn't cause violation but the second down causes a violation
						//In this case we insert a point on the first s down
						if (i + 1 == IA[act_row + 1] || JA[i + 1] != ss_up.first.second.first) {
							new_index = i + 1;
							updateJA(act_col, ss_up.first.second.second, new_index, bf.first[3], false);
							updateIA(act_row, act_row, bf.second[2], false);
							new_si = { bf.first[1], bf.first[2], bf.first[3], ss_up.second.second, ss_up.second.second };
						}
						else {
							//If no point on the first s up, then i+1 else i+2
							new_index = (getRowOfExisting(i + 1) == act_row && si_array[i + 1][2] <= bf.first[ss_up.second.first]) ? i + 2 : i + 1;
							updateJA(ss_up.first.second.first, ss_up.first.second.second, new_index, bf.first[ss_up.second.first], false);
							updateIA(act_row, act_row, bf.second[2], false);
							new_si = { bf.first[2], bf.first[3], bf.first[ss_up.second.first], ss_up.second.second, ss_up.second.second };
						}
					}

					auto ret_pair = insertAfterViol(new_index, new_si, bf.second, excluded, newlyAdded);
					excluded = ret_pair.first;
					newlyAdded = ret_pair.second;

					//Handle index changes due to new point inserted- to still check all points
					cpnum++;
					if (new_index <= i) i++;
				}

				//Recheck if needed -- could be solved better, less resource demanding
				if (needsOneMoreIter) j--;
			}
		}
	}
	std::pair<bool, std::pair<std::vector<int>, std::vector<int>>> ret_pair(violated, std::pair<std::vector<int>, std::vector<int>>(excluded,newlyAdded));
	return ret_pair;
}

void MyViewer::updateOrigs(double s, double t, int act_ind) {
	int first_orig = indsInOrig[act_ind - 1], sec_orig = indsInOrig[act_ind];

	//If they are in same orig row
	if (rowsInOrig[act_ind - 1] == rowsInOrig[act_ind]) {
		int temp_ind = first_orig + 1;
		while (origin_sarray[temp_ind][2] <= s) { temp_ind++; }
		temp_ind--;
		indsInOrig.emplace(indsInOrig.begin() + act_ind, temp_ind);
		rowsInOrig.emplace(rowsInOrig.begin() + act_ind, rowsInOrig[act_ind - 1]);
		colsInOrig.emplace(colsInOrig.begin() + act_ind, JAOrig[temp_ind]);
		if (distMode) {
			baseIndsInOrig.emplace(baseIndsInOrig.begin() + act_ind, temp_ind);
			fitDistances[temp_ind] = -1;
		}
	}
	else {
		//Finding the greatest row which has equal 
		int temp_row = rowsInOrig[act_ind - 1];
		int sec_row = rowsInOrig[act_ind];
		while (origin_tarray[IAOrig[temp_row]][2] <= t && temp_row <= sec_row) { temp_row++; }
		temp_row--;

		int temp_ind = IAOrig[temp_row];
		while (origin_sarray[temp_ind][2] <= s && temp_ind <= IAOrig[temp_row+1]) { temp_ind++; }
		temp_ind--;
		indsInOrig.emplace(indsInOrig.begin() + act_ind, temp_ind);
		rowsInOrig.emplace(rowsInOrig.begin() + act_ind, temp_row);
		colsInOrig.emplace(colsInOrig.begin() + act_ind, JAOrig[temp_ind]);
		if (distMode) {
			baseIndsInOrig.emplace(baseIndsInOrig.begin() + act_ind, temp_ind);
			fitDistances[temp_ind] = -1;
		}
		//TODO maybe handle case, where we have to go back one row
	}
}

std::pair<std::vector<int>, std::vector<int>> MyViewer::insertAfterViol(int new_index, std::vector<double> new_si, std::vector<double> new_ti, std::vector<int> excluded, std::vector<int> newlyAdded) {
	ti_array.insert(ti_array.begin() + new_index, new_ti);
	//Insert with new index into si_array - needs to be corrected anyway probably, so si of point i
	si_array.insert(si_array.begin() + new_index, new_si);
	std::pair<std::vector<double>, std::vector<double>> vec_pair(new_si, new_ti);
	std::vector<std::pair<std::vector<double>, std::vector<double>>> new_blend = { vec_pair };
	blend_functions.insert(blend_functions.begin() + new_index, new_blend);
	//New point can't influence others, so it doesn't need a good new_point and new_weight
	std::vector<Vec> new_point = { Vec() };
	refined_points.insert(refined_points.begin() + new_index, new_point);
	//initialize wieght with 0, so that it can be checked afterwards in order to delete it when updated with others
	std::vector<double> new_weight = { 0.0 };
	refined_weights.insert(refined_weights.begin() + new_index, new_weight);
	tspline_control_points.insert(tspline_control_points.begin() + new_index, new_point[0]);
	weights.insert(weights.begin() + new_index, new_weight[0]);

	//Update newlyAdded
	for (int na = 0; na < newlyAdded.size(); na++) {
		if (new_index <= newlyAdded[na]) newlyAdded[na]++;
	}
	newlyAdded.push_back(new_index);

	//Update excluded accordingly too
	for (int ex = 0; ex < excluded.size(); ex++) {
		if (excluded[ex] >= new_index) excluded[ex]++;
	}
	excluded.push_back(new_index);

	//Update edges temporarily with keeping old points but refreshing their indices - this way checkT/SUp/Down still functions correctly
	for (int e = 0; e < edges.size(); e++) {
		if (edges[e].first >= new_index) edges[e].first++;
		if (edges[e].second >= new_index) edges[e].second++;
	}

	if (bringBackMode || distMode) {
		updateOrigs(new_si[2],new_ti[2],new_index); 
	}

	std::pair<std::vector<int>, std::vector<int>> ret_pair(std::pair<std::vector<int>, std::vector<int>>(excluded, newlyAdded));
	return ret_pair;
}

void MyViewer::checkViolations(std::vector<int> excluded) {
	bool viol1 = false, viol2 = false;
	std::vector<int> newlyAdded = {excluded};
	do {
		auto viol1_pair = checkForViol1(excluded, newlyAdded);
		viol1 = viol1_pair.first;
		newlyAdded = viol1_pair.second.second;
		excluded = viol1_pair.second.first;
		std::pair<bool, std::pair<std::vector<int>, std::vector<int>>> viol2_pair = checkForViol2(excluded, newlyAdded);
		excluded = viol2_pair.second.first;
		newlyAdded = viol2_pair.second.second;
		viol2 = viol2_pair.first;
	} while (viol1 || viol2);

	
	int cpnum = tspline_control_points.size();
	for (int i = 0; i < cpnum; i++) {
		if (refined_points[i].size() != 0 || refined_weights[i].size() != 0 || blend_functions[i].size() != 0){
			//TODO Check if only one blend function for every point--if not-some error
		}
		//Do point and weight calculation here
		tspline_control_points[i] = refined_points[i][0];
		weights[i] = refined_weights[i][0];
		//Do updating si,tis based on blend_functions here
		si_array[i] = blend_functions[i][0].first;
		ti_array[i] = blend_functions[i][0].second;
	}
}

//Returns the two refined blend functions with the appropriate multipliers, first the one with c multiplier
std::pair<std::pair<double,std::vector<double>>, std::pair<double, std::vector<double>>> MyViewer::refineBlend(std::vector<double> knot_vector, int ins_ind, double new_value) {
	double c = ins_ind==4 ? 1.0 : (new_value - knot_vector[0]) / (knot_vector[3] - knot_vector[0]);
	double d = ins_ind == 1 ? 1.0 : (knot_vector[4] - new_value) / (knot_vector[4] - knot_vector[1]);
	std::vector<double> first_vec(knot_vector.begin(),knot_vector.begin()+4); //check:does this give back a 4 long vec??
	first_vec.insert(first_vec.begin()+ins_ind,new_value);
	std::vector<double> second_vec(knot_vector.begin()+1, knot_vector.begin() + 5);
	second_vec.insert(second_vec.begin() + ins_ind - 1, new_value);
	std::pair<double, std::vector<double>> first_pair(c,first_vec);
	std::pair<double, std::vector<double>> second_pair(d,second_vec);
	return std::pair<std::pair<double, std::vector<double>>, std::pair<double, std::vector<double>>>(first_pair,second_pair);
}

void MyViewer::insertRefined(double s, double t, int new_ind, int first_ind, int sec_ind) {
	//last paramater could be both false and true(?)
	updateIA(getRowOfExisting(first_ind), getRowOfExisting(sec_ind), t, false);
	//last paramater could be both false and true(?)
	updateJA(JA[first_ind], JA[sec_ind], new_ind, s, false);
	std::vector<double> new_ti = { 0, 0, t, 0, 0 };
	ti_array.insert(ti_array.begin() + new_ind, new_ti);
	std::vector<double> new_si = { 0, 0, s, 0, 0 };
	si_array.insert(si_array.begin() + new_ind, new_si);
	std::pair<std::vector<double>, std::vector<double>> vec_pair(new_si, new_ti);
	std::vector<std::pair<std::vector<double>, std::vector<double>>> new_blend = { vec_pair };
	blend_functions.insert(blend_functions.begin() + new_ind, new_blend);
	//New point can't influence others, so it doesn't need a good new_point and new_weight
	std::vector<Vec> new_point = { Vec() };
	refined_points.insert(refined_points.begin() + new_ind, new_point);
	//initialize wieght with 0, so that it can be checked afterwards in order to delete it when updated with others
	std::vector<double> new_weight = {0.0};
	refined_weights.insert(refined_weights.begin() + new_ind, new_weight);
	tspline_control_points.insert(tspline_control_points.begin() + new_ind, new_point[0]);
	weights.insert(weights.begin() + new_ind, new_weight[0]);
	std::vector<int> excluded = {new_ind};

	//Update edges temporarily with keeping old points but refreshing their indices - this way checkT/SUp/Down still functions correctly
	updateEdgesTemporarily(false, new_ind);
	//And replace the edge which was inserted on with the two new edges
	/*edges.erase(std::find(edges.begin(), edges.end(), std::pair<int, int>(first_ind, sec_ind+1)));
	edges.push_back(std::pair<int,int>(first_ind, new_ind));
	edges.push_back(std::pair<int, int>(new_ind, sec_ind+1));*/
	
	checkViolations(excluded);
	updateEdgeTopology();
	updateMesh();
	update();
}

//Checks whether there is an opposite t-junction in the epsilon neighbourhood, if so, it returns true and the value of opposite
std::pair<bool, double> MyViewer::checkOpposite(int act_row, int act_col, double s, double t, bool horizontal_insertion, int new_index, double epsilon) {
	std::vector<double> new_ti = { -1, -1, t, -1, -1 }; //TODO shift min to 0,0 in openTSpline - so no minus values will occur
	std::vector<double> new_si = { -1, -1, s, -1, -1 };
	std::vector<int> newlyAdded = {new_index};
	if (horizontal_insertion) {
		//Finding first t down according to Rule 1
		auto t_down = checkTsDown(act_row, act_col, new_index, new_si, new_ti, 0, newlyAdded); //second.first should be 1/first check should give error/, first should be false
		//Check if point with nearly same s exists
		int down_row = t_down.first.second.first;
		//Check all points in row for close point
		bool is_close = false;
		double close_value = -1.0;
		for (int i = IA[down_row]; i < IA[down_row + 1]; i++) {
			//WARNING edges and indices updated temporarily but ti and si not
			double temp_s = (i >= new_index) ? si_array[i-1][2] : si_array[i][2];
			if (abs(temp_s - s) < epsilon) {
				is_close = true;
				if (abs(temp_s - s) < abs(close_value - s)) close_value = temp_s;
			}
		}

		//Do the same upwards, only update close value if its closer than the closest downwards
		auto t_up = checkTsUp(act_row, act_col, new_index, new_si, new_ti, 0, newlyAdded); //second.first should be 1/first check should give error/, first should be false
		//Check if point with nearly same s exists
		int up_row = t_up.first.second.first; //first should give true, should be existing row
		//Check all points in row for close point
		for (int i = IA[up_row]; i < IA[up_row + 1]; i++) {
			//WARNING edges and indices updated temporarily but ti and si not
			double temp_s = (i >= new_index) ? si_array[i - 1][2] : si_array[i][2];
			if (abs(temp_s - s) < epsilon) {
				is_close = true;
				if (abs(temp_s - s) < abs(close_value - s)) close_value = temp_s;
			}
		}
		return { is_close,close_value };
	}
	else {
		//Finding first s down according to Rule 1
		auto s_down = checkSsDown(act_row, act_col, new_index, new_si, new_ti, 0, newlyAdded); //second.first should be 1/first check should give error/, first should be false
		//Check if point with nearly same t exists
		int down_col = s_down.first.second.first; //first should give true, should be existing col
		//Check all points in col for close point
		bool is_close = false;
		double close_value = -1.0;
		std::vector<int> col_indices = indicesOfColumn(down_col);
		for (auto i: col_indices) {
			//WARNING edges and indices updated temporarily but ti and si not
			double temp_t = (i >= new_index) ? ti_array[i - 1][2] : ti_array[i][2];
			if (abs(temp_t - t) < epsilon) {
				is_close = true;
				if (abs(temp_t - t) < abs(close_value - t)) close_value = temp_t;
			}
		}

		//Do the same upwards, only update close value if its closer than the closest downwards
		auto s_up = checkSsUp(act_row, act_col, new_index, new_si, new_ti, 0, newlyAdded); //second.first should be 1/first check should give error/, first should be false
		//Check if point with nearly same s exists
		int up_col = s_up.first.second.first; //first should give true, should be existing col
		col_indices = indicesOfColumn(up_col);
		//Check all points in col for close point
		for (auto i : col_indices) {
			//WARNING edges and indices updated temporarily but ti and si not
			double temp_t = (i >= new_index) ? ti_array[i - 1][2] : ti_array[i][2];
			if (abs(temp_t - t) < epsilon) {
				is_close = true;
				if (abs(temp_t - t) < abs(close_value - t)) close_value = temp_t;
			}
		}
		return { is_close,close_value };
	}
}

std::pair<std::vector<int>, std::vector<double>> MyViewer::refineRowCol(double new_value, int row_col_ind, bool is_row) {
	//Get the indices of points, whose blending function is going to be refined
	std::vector<int> indices;
	indices.clear();
	int new_ind; //index of insertion in the indices vector
	if (is_row) {
		int start_ind = IA[row_col_ind];
		int end_ind = IA[row_col_ind] - 1;
		//if(new_value <= si_array[start_ind][2] || new_value >= si_array[end_ind][2]) -- throw error
		for (int i = start_ind + 1; i <= end_ind; i++) {
			if (new_value < si_array[i][2]) {
				//If insertion after first in row, new_ind = 1
				new_ind = (i == start_ind + 1) ? 1 : 2;
				if (i - 2 >= start_ind)indices.push_back(i - 2);
				indices.push_back(i - 1);
				indices.push_back(i);
				if (i + 1 <= end_ind)indices.push_back(i + 1);
			}
		}
	}
	else {
		std::vector<int> col_inds = indicesOfColumn(row_col_ind);
		//if(new_value <= ti_array[col_inds[0]][2] || new_value >= ti_array[col_inds[col_inds.size()-1]][2]) -- throw error
		for (int i = 1; i < col_inds.size(); i++) {
			if (new_value < ti_array[i][2]) {
				//If insertion after first in col, new_ind = 1
				new_ind = (i == 1) ? 1 : 2;
				if (i >= 2)indices.push_back(col_inds[i - 2]);
				indices.push_back(col_inds[i - 1]);
				indices.push_back(col_inds[i]);
				if (i + 1 < col_inds.size())indices.push_back(col_inds[i + 1]);
			}
		}
	}

	//Calculate multipliers
	std::vector<double> multipliers;
	multipliers.resize(indices.size() + 1);
	for (auto i : multipliers) i = 0.0;
	std::vector<double> knot_vector;
	for (int i = 0; i < indices.size(); i++) {
		// knot_vector.clear();
		knot_vector = (is_row) ? si_array[indices[i]] : ti_array[indices[i]];
		//First in indices
		if (i == 0) {
			if (new_ind == 2) {
				multipliers[0] += 1.0;
				multipliers[1] += (knot_vector[4] - new_value) / (knot_vector[4] - knot_vector[1]);
			}
			else {
				multipliers[0] += (new_value - knot_vector[0]) / (knot_vector[3] - knot_vector[0]);
				multipliers[1] += (knot_vector[4] - new_value) / (knot_vector[4] - knot_vector[1]);
			}
		}
		//Last in indices
		else if (i == indices.size() - 1) {
			if (i - new_ind == 2) {
				multipliers[i] += (new_value - knot_vector[0]) / (knot_vector[3] - knot_vector[0]);
				multipliers[i + 1] += 1.0;
			}
			else {
				multipliers[i] += (new_value - knot_vector[0]) / (knot_vector[3] - knot_vector[0]);
				multipliers[i + 1] += (knot_vector[4] - new_value) / (knot_vector[4] - knot_vector[1]);
			}
		}
		else {
			if (i < new_ind) {
				multipliers[i] += (new_value - knot_vector[0]) / (knot_vector[3] - knot_vector[0]);
				multipliers[i + 1] += (knot_vector[4] - new_value) / (knot_vector[4] - knot_vector[1]);
			}
			else {
				multipliers[i - 1] += (new_value - knot_vector[0]) / (knot_vector[3] - knot_vector[0]);
				multipliers[i] += (knot_vector[4] - new_value) / (knot_vector[4] - knot_vector[1]);
			}
		}
	}

	indices.insert(indices.begin() + new_ind, -1);
	std::pair<std::vector<int>, std::vector<double>> ret_pair;
	ret_pair.first = indices;
	ret_pair.second = multipliers;
	return ret_pair;
}

void MyViewer::postSelection(const QPoint &p)  {
  int sel = selectedName();
  if (sel == -1) {
    axes.shown = false;
    return;
  }

  if (axes.shown) {
    axes.selected_axis = sel;
    bool found;
    axes.grabbed_pos = camera()->pointUnderPixel(p, found);
    axes.original_pos = axes.position;
    if (!found)
      axes.shown = false;
    return;
  }
  bool edge = false;
  int cpnum = tspline_control_points.size();
  selected_vertex = sel;
  if (model_type == ModelType::MESH)
    axes.position = Vec(mesh.point(MyMesh::VertexHandle(sel)).data());
  if (model_type == ModelType::BEZIER_SURFACE)
    axes.position = bezier_control_points[sel];
  if (model_type == ModelType::TSPLINE_SURFACE)
	  if (sel >= cpnum) edge = true;
	  else axes.position = tspline_control_points[sel]/weights[sel];
  if (!edge) {
	  double depth = camera()->projectedCoordinatesOf(axes.position)[2];
	  Vec q1 = camera()->unprojectedCoordinatesOf(Vec(0.0, 0.0, depth));
	  Vec q2 = camera()->unprojectedCoordinatesOf(Vec(width(), height(), depth));
	  axes.size = (q1 - q2).norm() / 10.0;
	  axes.shown = true;
	  axes.selected_axis = -1;
  }
  else {
	  double epsilon = 0.05;

	  std::pair<int, int> index_pair = edges[sel - cpnum];
	  double alpha = 0.5, beta = 0.5;
	  Vec selectedPoint;
	  if(mid_insert){
		  selectedPoint = ((tspline_control_points[index_pair.first] / weights[index_pair.first]) + (tspline_control_points[index_pair.second] / weights[index_pair.second])) / 2.0;
	  }
	  else {
		  //Select point under pixel
		  bool found;
		  selectedPoint = camera()->pointUnderPixel(p, found);
		  if (!found) return;
		  alpha = (selectedPoint - tspline_control_points[index_pair.first]/weights[index_pair.first]).norm() / (tspline_control_points[index_pair.second] / weights[index_pair.second] - tspline_control_points[index_pair.first] / weights[index_pair.first]).norm();
	  }
	  beta = (alpha * weights[index_pair.first]) / (alpha * weights[index_pair.first] + (1.0 - alpha) * weights[index_pair.second]);

	  std::vector<double> new_si, new_ti;
	  int new_index;
	  double new_s, new_t;
	  int rowOfFirst = getRowOfExisting(index_pair.first);
	  int rowOfSecond = getRowOfExisting(index_pair.second);
	  //If in same row, otherwise they must be in same column
	  if (rowOfFirst == rowOfSecond) {
		  //Calculating beta based on refinement of blending functions - (new_s - s1)/(s4-s1) = beta
		  new_s = beta * (si_array[index_pair.first][4] - si_array[index_pair.first][1]) + si_array[index_pair.first][1];

		  while (new_s > si_array[index_pair.second][2]) {
			  index_pair.first++;
			  index_pair.second++;
		  }
		  while (new_s < si_array[index_pair.first][2]) {
			  index_pair.first--;
			  index_pair.second--;
		  }
		  new_t = ti_array[index_pair.first][2];
		  new_index = index_pair.second;

		  int act_row;

		  if (!mid_insert) {
			  //Update JA and IA temporarily
			  updateJA(JA[index_pair.first], JA[index_pair.second], new_index, new_s, false);  //last parameter could be both(?)
			  updateIA(getRowOfExisting(index_pair.first), getRowOfExisting(index_pair.second), new_t, false); //last parameter could be both(?)
			  updateEdgesTemporarily(false, new_index);

			  act_row = getRowOfExisting(new_index);
			  int act_col = JA[new_index];
			  auto opp_check = checkOpposite(act_row, act_col, new_s, new_t, true, new_index, epsilon);
			  if (opp_check.first) new_s = opp_check.second;
			  deleteFromJA(new_index);
			  deleteFromIA(new_index);
			  updateEdgesTemporarily(true, new_index);
		  }
		  else {
			  act_row = getRowOfExisting(index_pair.first);
		  }

		  //TODO visual feedback for changing keep_surface
		  if (keep_surface) {
			  insertRefined(new_s, new_t, new_index, index_pair.first, index_pair.second);
			  return;
		  }

		  new_si = { si_array[index_pair.first][1], si_array[index_pair.first][2], new_s, si_array[index_pair.second][2], si_array[index_pair.second][3]};
		  

		  //Finding new ti
		  new_ti.clear();
		  

		  //Check t-s downwards
		  int i = act_row;
		  int temp_ind = i == 0 ? 0 : IA[--i]; //start index of row (of first)-1
		  int num_found = 0;
		  while (num_found < 2) {
			  if (temp_ind == 0) {
				  new_ti.insert(new_ti.begin(), ti_array[0][2]);
				  num_found++;
			  }
			  else {
				  for (; si_array[temp_ind][2] <= new_s && getRowOfExisting(temp_ind) == i; temp_ind++) {
				  }
				  if (si_array[temp_ind-1][2] < new_s){
					  //Check if not the case of last in row having smaller s than new_s
					  if (si_array[temp_ind][2] > new_s) {
						  //check whether there is an edge connecting temp_ind-1 and temp_ind,
						  //meaning that a vertical ray started from the new point would cut it,
						  //and so the t of them should be stored in new_ti
						  bool found = false;
						  for (int j = 0; j < edges.size(), !found; j++) {
							  auto p = edges[j];
							  if ((p.first == temp_ind - 1) && (p.second == temp_ind)) {
								  new_ti.insert(new_ti.begin(), ti_array[temp_ind - 1][2]);
								  num_found++;
								  found = true;
							  }
						  }
					  }
				  } //First of actual row has greater s than our point
				  else if (i != getRowOfExisting(temp_ind - 1)) {}
				  else {
					  //This case occurs when si_array[temp_ind - 1][2] == new_s
					  new_ti.insert(new_ti.begin(), ti_array[temp_ind-1][2]);
					  num_found++;
				  }
				  temp_ind = IA[--i];
			  }
		  }

		  new_ti.push_back(new_t);

		  //Check t-s upwards
		  i = act_row;
		  temp_ind = i == IA.size()-2 ? IA[IA.size()-2] : IA[++i]; //start index of row (of first)+1
		  num_found = 0;
		  while (num_found < 2) {
			  if (temp_ind == IA[IA.size()-2]) {
				  new_ti.push_back(ti_array[cpnum-1][2]);
				  num_found++;
			  }
			  else {
				  for (; si_array[temp_ind][2] <= new_s && getRowOfExisting(temp_ind) == i; temp_ind++) {
				  }
				  if (si_array[temp_ind - 1][2] < new_s) {
					  //Check if not the case of last in row having smaller s than new_s
					  if (si_array[temp_ind][2] > new_s) {
						  //check whether there is an edge connecting temp_ind-1 and temp_ind,
						  //meaning that a vertical ray started from the new point would cut it,
						  //and so the t of them should be stored in new_ti
						  bool found = false;
						  for (int j = 0; j < edges.size(), !found; j++) {
							  auto p = edges[j];
							  if ((p.first == temp_ind - 1) && (p.second == temp_ind)) {
								  new_ti.push_back(ti_array[temp_ind - 1][2]);
								  num_found++;
								  found = true;
							  }
						  }
					  }
				  } //First of actual row has greater s than our point
				  else if (i != getRowOfExisting(temp_ind - 1)) {}
				  else {
					  //This case occurs when si_array[temp_ind - 1][2] == new_s
					  new_ti.push_back(ti_array[temp_ind - 1][2]);
					  num_found++;
				  }
				  temp_ind = IA[++i];
			  }
		  }
		  //Update JA
		  updateJA(JA[index_pair.first], JA[index_pair.second], new_index, new_s, false); //last parameter could be both(?)
		  //Update neighbouring si-s
		  si_array[index_pair.first][3] = new_s;
		  si_array[index_pair.first][4] = si_array[index_pair.second][2];
		  if (index_pair.first != 0 && ti_array[index_pair.first][2] == ti_array[index_pair.first - 1][2]) {
			  si_array[index_pair.first-1][3] = si_array[index_pair.first][2];
			  si_array[index_pair.first-1][4] = new_s;
		  }
		  si_array[index_pair.second][1] = new_s;
		  si_array[index_pair.second][0] = si_array[index_pair.first][2];
		  if (index_pair.second != cpnum-1 && ti_array[index_pair.second][2] == ti_array[index_pair.second + 1][2]) {
			  si_array[index_pair.second + 1][1] = si_array[index_pair.second][2];
			  si_array[index_pair.second + 1][0] = new_s;
		  }

		  //Updating IA and JA matrices too
		  //Update IA
		  for (int i = act_row + 1; i < IA.size(); i++) {
			  IA[i] += 1;
		  }
	  }
	  else{
		auto col_inds = indicesOfColumn(JA[index_pair.first]);
		//Calculating beta based on refinement of blending functions - (new_t - t1)/(t4-t1) = beta
		new_t = beta * (ti_array[index_pair.first][4] - ti_array[index_pair.first][1]) + ti_array[index_pair.first][1];

		int indOfSecInCol = std::find(col_inds.begin(), col_inds.end(), index_pair.second) - col_inds.begin();
		int indOfFirstInCol = indOfSecInCol - 1;
		new_s = si_array[index_pair.first][2];
		
		while (new_t > ti_array[index_pair.second][2]) {
			index_pair.first = index_pair.second;
			index_pair.second = col_inds[++indOfSecInCol];
		}
		while (new_t < ti_array[index_pair.first][2]) {
			//Case when first in col has greater t than 0 but new_t is lower than this t
			//We insert a point with t = 0 and then another one on the edge connecting these two
			/*if (indOfFirstInCol == 0 && keep_surface) {
				int addInd = getIndex(1, 1, JA[index_pair.first], 0.0, true);
				insertRefined(new_s,0.0,addInd,addInd-1,addInd);
			}*/
			index_pair.second = index_pair.first;
			index_pair.first = col_inds[--indOfFirstInCol];
		}
		  
		//Finding new index
		new_index = getIndex(getRowOfExisting(index_pair.first), getRowOfExisting(index_pair.second), JA[index_pair.first], new_t, false); //last parameter could be both(?)
		  
		int act_col;
		if (!mid_insert) {
			//Update JA and IA temporarily
			updateJA(JA[index_pair.first], JA[index_pair.second], new_index, new_s, false); //last parameter could be both(?)
			updateIA(getRowOfExisting(index_pair.first), getRowOfExisting(index_pair.second), new_t, false); //last parameter could be both(?)
			updateEdgesTemporarily(false, new_index);

			int act_row = getRowOfExisting(new_index);
			act_col = JA[new_index];
			auto opp_check = checkOpposite(act_row, act_col, new_s, new_t, false, new_index, epsilon);
			deleteFromJA(new_index);
			deleteFromIA(new_index);
			updateEdgesTemporarily(true, new_index);

			if (opp_check.first) {
				new_t = opp_check.second;
				//Updating index
				new_index = getIndex(getRowOfExisting(index_pair.first), getRowOfExisting(index_pair.second), JA[index_pair.first], new_t, false); //last parameter could be both(?)
			}
		}

		if (keep_surface) {
			insertRefined(new_s, new_t,new_index, index_pair.first, index_pair.second);
			return;
		}

		new_ti = { ti_array[index_pair.first][1], ti_array[index_pair.first][2], new_t, ti_array[index_pair.second][2], ti_array[index_pair.second][3] };
		  
		  

		//Finding new si
		new_si.clear();
		int col_num = *std::max_element(JA.begin(), JA.end()) + 1;
		act_col = JA[index_pair.first];

		//Check s-s downwards
		int i = act_col==0 ? act_col : act_col-1;
		int num_found = 0;
		while (num_found < 2) {
			if (i == 0) {
				new_si.insert(new_si.begin(),si_array[0][2]);
				num_found++;
			}
			else {
				std::vector<int> is_of_col = indicesOfColumn(i);
				int j = 0;
				for (; j < is_of_col.size() && ti_array[is_of_col[j]][2] <= new_t; j++) {
				}
				//If first in act_col has bigger t than new_t
				if (j == 0) {}
				if (ti_array[is_of_col[j-1]][2] < new_t) {
					//Check if not the case of last in col having smaller t than new_t
					if (ti_array[is_of_col[j]][2] > new_t) {
						//check whether there is an edge connecting temp_ind-1 and temp_ind,
						//meaning that a vertical ray started from the new point would cut it,
						//and so the t of them should be stored in new_ti
						bool found = false;
						for (int k = 0; k < edges.size(), !found; k++) {
							auto p = edges[k];
							if ((p.first == is_of_col[j - 1]) && (p.second == is_of_col[j])) {
								new_si.insert(new_si.begin(), si_array[is_of_col[j - 1]][2]);
								num_found++;
								found = true;
							}
						}
					}
				}
				else {
					//This case occurs when ti_array[is_of_col[j-1]][2] == new_t
					new_si.insert(new_si.begin(), si_array[is_of_col[j - 1]][2]);
					num_found++;
				}
				--i;
			}
		}

		new_si.push_back(new_s);

		//Check s-s upwards
		i = act_col == col_num-1 ? col_num-1 : act_col + 1;
		num_found = 0;
		while (num_found < 2) {
			if (i == col_num-1) {
				new_si.push_back(si_array[cpnum-1][2]);
				num_found++;
			}
			else {
				std::vector<int> is_of_col = indicesOfColumn(i);
				int j = 0;
				for (; j < is_of_col.size() && ti_array[is_of_col[j]][2] <= new_t; j++) {
				}
				//If first in act_col has bigger t than new_t
				if (j == 0) {}
				if (ti_array[is_of_col[j - 1]][2] < new_t) {
					//Check if not the case of last in col having smaller t than new_t
					if (ti_array[is_of_col[j]][2] > new_t) {
						//check whether there is an edge connecting temp_ind-1 and temp_ind,
						//meaning that a vertical ray started from the new point would cut it,
						//and so the t of them should be stored in new_ti
						bool found = false;
						for (int k = 0; k < edges.size(), !found; k++) {
							auto p = edges[k];
							if ((p.first == is_of_col[j - 1]) && (p.second == is_of_col[j])) {
								new_si.push_back(si_array[is_of_col[j - 1]][2]);
								num_found++;
								found = true;
							}
						}
					}
				}
				else {
					//This case occurs when ti_array[is_of_col[j-1]][2] == new_t
					new_si.push_back(si_array[is_of_col[j - 1]][2]);
					num_found++;
				}
				++i;
			}
		}

		//Updating IA and JA matrices too
		//Update JA
		JA.insert(JA.begin() + new_index, act_col);
		//Update IA too
		updateIA(getRowOfExisting(index_pair.first), getRowOfExisting(index_pair.second), new_t, false); //last parameter could be both(?)

		//Update neighbouring ti-s
		ti_array[index_pair.first][3] = new_t;
		ti_array[index_pair.first][4] = ti_array[index_pair.second][2];
		auto ts_of_actcol = indicesOfColumn(act_col);
		//Find the index of index_pair.first in its column
		bool found = false;
		int k = 0;
		for (; k < ts_of_actcol.size(), !found;k++) {
			if (ts_of_actcol[k] == index_pair.first) {
				found = true;
				k--;
			}
		}
		if (k != 0) {
			ti_array[ts_of_actcol[k-1]][3] = ti_array[ts_of_actcol[k]][2];
			ti_array[ts_of_actcol[k-1]][4] = new_t;
		}
		ti_array[index_pair.second][1] = new_t;
		ti_array[index_pair.second][0] = ti_array[index_pair.first][2];
		if (k+1 != ts_of_actcol.size()-1) {
			ti_array[ts_of_actcol[k+2]][1] = ti_array[ts_of_actcol[k+1]][2];
			ti_array[ts_of_actcol[k+2]][0] = new_t;
		}

		  
	  }
	  double weight = 1.0; //TODO other weight value??
	  weights.insert(weights.begin() + new_index, weight);
	  si_array.insert(si_array.begin() + new_index, new_si);
	  ti_array.insert(ti_array.begin() + new_index, new_ti);
	  tspline_control_points.insert(tspline_control_points.begin() + new_index, selectedPoint);
	  updateEdgeTopology();
	  updateMesh();
	  update();
  }
}

Vec MyViewer::intersectLines(const Vec &ap, const Vec &ad, const Vec &bp, const Vec &bd) {
  // always returns a point on the (ap, ad) line
  double a = ad * ad, b = ad * bd, c = bd * bd;
  double d = ad * (ap - bp), e = bd * (ap - bp);
  if (a * c - b * b < 1.0e-7)
    return ap;
  double s = (b * e - c * d) / (a * c - b * b);
  return ap + s * ad;
}

void MyViewer::bernsteinAll(size_t n, double u, std::vector<double> &coeff) {
  coeff.clear(); coeff.reserve(n + 1);
  coeff.push_back(1.0);
  double u1 = 1.0 - u;
  for (size_t j = 1; j <= n; ++j) {
    double saved = 0.0;
    for (size_t k = 0; k < j; ++k) {
      double tmp = coeff[k];
      coeff[k] = saved + tmp * u1;
      saved = tmp * u;
    }
    coeff.push_back(saved);
  }
}

double MyViewer::cubicBSplineBasis(double param, std::vector<double> knots) {
	double u = param;
	int p = 3, i;
	int end = knots.size()-2;
	for (int j = knots.size()-2; j >= 0; j--) {
		if (knots[j] >= knots[end+1]) {
			end = j-1;
		}
		else break;
	}

	if (u < knots.front() || u > knots.back())
		return 0.0;
	if (u == knots.back())
		i = end;
	else i = (std::upper_bound(knots.begin(), knots.end(), u) - knots.begin()) - 1;
	std::vector<double> coeff;
	coeff.resize(p+1, 0.0);
	coeff[i] = 1.0;

	for (int j = 1; j <= p; ++j)
		for (int k = 0; k < (knots.size()-1) - j; ++k)
			coeff[k] =
			(coeff[k] && (knots[k + j] - knots[k]) != 0.0 ? coeff[k] * (u - knots[k]) / (knots[k + j] - knots[k]) : 0.0) +
			(coeff[k + 1] && (knots[k + j + 1] - knots[k + 1]) != 0.0 ? coeff[k + 1] * (knots[k + j + 1] - u) / (knots[k + j + 1] - knots[k + 1]) : 0.0);
	return coeff[0];
}

void MyViewer::generateBezierMesh() {
  size_t resolution = 30;

  mesh.clear();
  std::vector<MyMesh::VertexHandle> handles, tri;
  size_t n = degree[0], m = degree[1];

  std::vector<double> coeff_u, coeff_v;
  for (size_t i = 0; i < resolution; ++i) {
    double u = (double)i / (double)(resolution - 1);
    bernsteinAll(n, u, coeff_u);
    for (size_t j = 0; j < resolution; ++j) {
      double v = (double)j / (double)(resolution - 1);
      bernsteinAll(m, v, coeff_v);
      Vec p(0.0, 0.0, 0.0);
      for (size_t k = 0, index = 0; k <= n; ++k)
        for (size_t l = 0; l <= m; ++l, ++index)
          p += bezier_control_points[index] * coeff_u[k] * coeff_v[l];
      handles.push_back(mesh.add_vertex(Vector(static_cast<double *>(p))));
    }
  }
  for (size_t i = 0; i < resolution - 1; ++i)
    for (size_t j = 0; j < resolution - 1; ++j) {
      tri.clear();
      tri.push_back(handles[i * resolution + j]);
      tri.push_back(handles[i * resolution + j + 1]);
      tri.push_back(handles[(i + 1) * resolution + j]);
      mesh.add_face(tri);
      tri.clear();
      tri.push_back(handles[(i + 1) * resolution + j]);
      tri.push_back(handles[i * resolution + j + 1]);
      tri.push_back(handles[(i + 1) * resolution + j + 1]);
      mesh.add_face(tri);
    }
}

void MyViewer::generateTSplineMesh() {
	size_t resolution = 50;
	size_t cpnum = weights.size();

	mesh.clear();
	std::vector<MyMesh::VertexHandle> handles, tri;
	for (size_t i = 0; i < resolution; ++i) {
		double s = (double)i / (double)(resolution - 1);
		for (size_t j = 0; j < resolution; ++j) {
			double t = (double)j / (double)(resolution - 1);
			Vec p(0.0, 0.0, 0.0);
			double nominator = 0.0;
			for (size_t k = 0; k < cpnum; ++k) {
				double B_k = cubicBSplineBasis(s,si_array[k]) * cubicBSplineBasis(t, ti_array[k]);
				p += tspline_control_points[k] * B_k;
				nominator += weights[k] * B_k;
			}
			if(abs(nominator) > 0.0) p /= nominator;
			handles.push_back(mesh.add_vertex(Vector(static_cast<double *>(p))));
		}
	}
	for (size_t i = 0; i < resolution - 1; ++i)
		for (size_t j = 0; j < resolution - 1; ++j) {
			tri.clear();
			tri.push_back(handles[i * resolution + j]);
			tri.push_back(handles[i * resolution + j + 1]);
			tri.push_back(handles[(i + 1) * resolution + j]);
			mesh.add_face(tri);
			tri.clear();
			tri.push_back(handles[(i + 1) * resolution + j]);
			tri.push_back(handles[i * resolution + j + 1]);
			tri.push_back(handles[(i + 1) * resolution + j + 1]);
			mesh.add_face(tri);
		}
}

//TODO moving point after multiple insertion made an error in surface - right side disappeared

//TODO organize postSelection blocks into these 4 checkfunctions

//TODO make the return of check functions prettier

/*
Returns with
first.first: false if the check found an error,true if not
first.second.first: index of first found row
first.second.second: index of second found row
second.first: index of insertion in t_vec
second.second: new value to be inserted
*/
std::pair<std::pair<bool, std::pair<int,int>>,std::pair<int,double>> MyViewer::checkTsDown(int act_row, int act_col, int index, std::vector<double> s_vec, std::vector<double> t_vec, int viol_num, std::vector<int> excluded) {
	int temp_ind = act_row == 0 ? 0 : IA[--act_row]; //start index of row(of index)+1
	int num_found = 0;
	std::pair<int, int> ret_rows = {-1, -1};
	while (num_found < 2) {
		//Check whether actual row is the first one
		if (temp_ind == 0) {
			if(num_found == 0) ret_rows.first = act_row;
			else ret_rows.second = act_row;
			if (t_vec[1 - num_found] != ti_array[0][2]) {
				//If checking for viol 1 but before that a viol2 occurs
				//In this case we should return with true(?)
				if (viol_num == 1 && t_vec[1 - num_found] > ti_array[0][2])
					return std::pair<std::pair<bool, std::pair<int, int>>, std::pair<int, double>>(std::pair<bool, std::pair<int, int>>(true, ret_rows), std::pair<int, double>(-1, 0.0));
				if ((viol_num == 1 && t_vec[1 - num_found] < ti_array[0][2])
					|| (viol_num == 2 && t_vec[1 - num_found] > ti_array[0][2])
					|| viol_num == 0)
					return std::pair<std::pair<bool, std::pair<int, int>>, std::pair<int, double>>(std::pair<bool, std::pair<int, int>>(false, ret_rows), std::pair < int, double>(1 - num_found, ti_array[0][2]));
			}
			num_found++;
		}
		else {
			for (; si_array[temp_ind][2] <= s_vec[2] && getRowOfExisting(temp_ind) == act_row; temp_ind++) {
			}
			if (si_array[temp_ind - 1][2] < s_vec[2]) {
				//Check if not the case of last in row having smaller s than the point with index "index"
				if (si_array[temp_ind][2] > s_vec[2]) {
					//check whether there is an edge connecting temp_ind-1 and temp_ind,
					//meaning that a vertical ray started from our point would cut it,
					//and so the t of them should be the 1-num_found-th element of ti array of our point
					bool found = false;
					int first_tmp = temp_ind - 1;
					int sec_tmp = temp_ind;
					//If one of temp_ind or temp_ind -1 is in excluded, we must check the next index accordingly
					//in order to keep the checking correct by checking only edges which were originally included
					while (getRowOfExisting(first_tmp) == act_row && std::find(excluded.begin(), excluded.end(), first_tmp) != excluded.end()) {
						first_tmp--;
					}
					while (getRowOfExisting(sec_tmp) == act_row && std::find(excluded.begin(), excluded.end(), sec_tmp) != excluded.end()) {
						sec_tmp++;
					}

					for (int j = 0; j < edges.size() && !found; j++) {
						auto p = edges[j];
						if ((p.first == first_tmp) && (p.second == sec_tmp)) {
							if (num_found == 0) ret_rows.first = act_row;
							else ret_rows.second = act_row;
							if (t_vec[1 - num_found] != ti_array[first_tmp][2]) {
								//If checking for viol 1 but before that a viol2 occurs
								//In this case we should return with true(?)
								if (viol_num == 1 && t_vec[1 - num_found] > ti_array[first_tmp][2])
									return std::pair<std::pair<bool, std::pair<int, int>>, std::pair<int, double>>(std::pair<bool, std::pair<int, int>>(true, ret_rows), std::pair<int, double>(-1, 0.0));
								if ((viol_num == 1 && t_vec[1 - num_found] < ti_array[first_tmp][2])
									|| (viol_num == 2 && t_vec[1 - num_found] > ti_array[first_tmp][2])
									|| viol_num == 0)
									return std::pair<std::pair<bool, std::pair<int, int>>, std::pair<int, double>>(std::pair<bool, std::pair<int, int>>(false, ret_rows), std::pair<int, double>(1 - num_found, ti_array[first_tmp][2]));
							}
							num_found++;
							found = true;
						}
					}
				}
			} //First of actual row has greater s than our point or the biggest of cols with same s in act_row is earlier than ours
			else if (act_row != getRowOfExisting(temp_ind - 1) || JA[index] > JA[temp_ind - 1]) {}
			else {
				//This case occurs when si_array[temp_ind - 1][2] == s_vec[2]
				while (getRowOfExisting(temp_ind - 1) == act_row && act_col != JA[temp_ind - 1]) {
					temp_ind--;
				}
				//!= if first in act_row has same s, but is in a col with greater index
				if (getRowOfExisting(temp_ind - 1) == act_row) {
					if (num_found == 0) ret_rows.first = act_row;
					else ret_rows.second = act_row;
					if (t_vec[1 - num_found] != ti_array[temp_ind - 1][2]) {
						//If checking for viol 1 but before that a viol2 occurs
						//In this case we should return with true(?)
						if (viol_num == 1 && t_vec[1 - num_found] > ti_array[temp_ind - 1][2])
							return std::pair<std::pair<bool, std::pair<int, int>>, std::pair<int, double>>(std::pair<bool, std::pair<int, int>>(true, ret_rows), std::pair<int, double>(-1, 0.0));
						if ((viol_num == 1 && t_vec[1 - num_found] < ti_array[temp_ind - 1][2])
							|| (viol_num == 2 && t_vec[1 - num_found] > ti_array[temp_ind - 1][2])
							|| viol_num == 0)
							return std::pair<std::pair<bool, std::pair<int, int>>, std::pair<int, double>>(std::pair<bool, std::pair<int, int>>(false, ret_rows), std::pair<int, double>(1 - num_found, ti_array[temp_ind - 1][2]));
					}
					num_found++;
				}
			}
			temp_ind = IA[--act_row];
		}
	}
	return std::pair<std::pair<bool, std::pair<int, int>>, std::pair<int, double>>(std::pair<bool, std::pair<int, int>>(true, ret_rows), std::pair<int, double>(-1,0.0));
}

/*
Returns with
first.first: false if the check found an error,true if not
first.second.first: index of first found row
first.second.second: index of second found row
second.first: index of insertion in t_vec
second.second: new value to be inserted
*/
std::pair<std::pair<bool, std::pair<int, int>>, std::pair<int, double>> MyViewer::checkTsUp(int act_row, int act_col, int index, std::vector<double> s_vec, std::vector<double> t_vec, int viol_num, std::vector<int> excluded) {
	int temp_ind = act_row == IA.size() - 2 ? IA[IA.size() - 2] : IA[++act_row]; //start index of row(of index)+1
	int num_found = 0;
	int cpnum = tspline_control_points.size();
	std::pair<int, int> ret_rows = { -1, -1 };
	while (num_found < 2) {
		//Check whether actual row is the last one
		if (temp_ind == IA[IA.size() - 2]) {
			if (num_found == 0) ret_rows.first = act_row;
			else ret_rows.second = act_row;
			if (t_vec[3 + num_found] != ti_array[cpnum - 1][2]) {
				//If checking for viol 1 but before that a viol2 occurs
				//In this case we should return with true(?)
				if (viol_num == 1 && t_vec[3 + num_found] < ti_array[cpnum - 1][2])
					return std::pair<std::pair<bool, std::pair<int, int>>, std::pair<int, double>>(std::pair<bool, std::pair<int, int>>(true, ret_rows), std::pair<int, double>(-1, 0.0));
				if ((viol_num == 1 && t_vec[3 + num_found] > ti_array[cpnum - 1][2])
					|| (viol_num == 2 && t_vec[3 + num_found] < ti_array[cpnum - 1][2])
					|| viol_num == 0)
					return std::pair<std::pair<bool, std::pair<int, int>>, std::pair<int, double>>(std::pair<bool, std::pair<int, int>>(false, ret_rows), std::pair < int, double>(3 + num_found, ti_array[cpnum - 1][2]));
			}
			num_found++;
		}
		else {
			for (; si_array[temp_ind][2] <= s_vec[2] && getRowOfExisting(temp_ind) == act_row; temp_ind++) {
			}
			if (si_array[temp_ind - 1][2] < s_vec[2]) {
				//Check if not the case of last in row having smaller s than our point
				if (si_array[temp_ind][2] > s_vec[2]) {
					//Check whether there is an edge connecting temp_ind-1 and temp_ind,
					//meaning that a vertical ray started from our point would cut it,
					//and so the t of them should be the 3+num_found-th element of ti array of our point
					bool found = false;
					int first_tmp = temp_ind - 1;
					int sec_tmp = temp_ind;
					//If one of temp_ind or temp_ind -1 is in excluded, we must check the next index accordingly
					//in order to keep the checking correct by checking only edges which were originally included
					while (getRowOfExisting(first_tmp) == act_row && std::find(excluded.begin(), excluded.end(), first_tmp) != excluded.end()) {
						first_tmp--;
					}
					while (getRowOfExisting(sec_tmp) == act_row && std::find(excluded.begin(), excluded.end(), sec_tmp) != excluded.end()) {
						sec_tmp++;
					}
					for (int j = 0; j < edges.size() && !found; j++) {
						auto p = edges[j];
						if ((p.first == first_tmp) && (p.second == sec_tmp)) {
							if (num_found == 0) ret_rows.first = act_row;
							else ret_rows.second = act_row;
							if (t_vec[3 + num_found] != ti_array[first_tmp][2]) {
								//If checking for viol 1 but before that a viol2 occurs
								//In this case we should return with true(?)
								if (viol_num == 1 && t_vec[3 + num_found] < ti_array[first_tmp][2])
									return std::pair<std::pair<bool, std::pair<int, int>>, std::pair<int, double>>(std::pair<bool, std::pair<int, int>>(true, ret_rows), std::pair<int, double>(-1, 0.0));
								if ((viol_num == 1 && t_vec[3 + num_found] > ti_array[first_tmp][2])
									|| (viol_num == 2 && t_vec[3 + num_found] < ti_array[first_tmp][2])
									|| viol_num == 0)
									return std::pair<std::pair<bool, std::pair<int, int>>, std::pair<int, double>>(std::pair<bool, std::pair<int, int>>(false, ret_rows), std::pair<int, double>(3 + num_found, ti_array[first_tmp][2]));
							}
							num_found++;
							found = true;
						}
					}
				}
			} //First of actual row has greater s than our point or the biggest of cols with same s in act_row is earlier than ours
			else if(act_row != getRowOfExisting(temp_ind-1) || JA[index] > JA[temp_ind - 1]){}
			else {
				//This case occurs when si_array[temp_ind - 1][2] == s_vec[2]
				while (getRowOfExisting(temp_ind - 1) == act_row && act_col != JA[temp_ind - 1]) {
					temp_ind--;
				}
				//!= if first in act_row has same s, but is in a col with greater index
				if (getRowOfExisting(temp_ind - 1) == act_row) {
					if (num_found == 0) ret_rows.first = act_row;
					else ret_rows.second = act_row;
					if (t_vec[3 + num_found] != ti_array[temp_ind - 1][2]) {
						//If checking for viol 1 but before that a viol2 occurs
						//In this case we should return with true(?)
						if (viol_num == 1 && t_vec[3 + num_found] < ti_array[temp_ind - 1][2])
							return std::pair<std::pair<bool, std::pair<int, int>>, std::pair<int, double>>(std::pair<bool, std::pair<int, int>>(true, ret_rows), std::pair<int, double>(-1, 0.0));
						if ((viol_num == 1 && t_vec[3 + num_found] > ti_array[temp_ind - 1][2])
							|| (viol_num == 2 && t_vec[3 + num_found] < ti_array[temp_ind - 1][2])
							|| viol_num == 0)
							return std::pair<std::pair<bool, std::pair<int, int>>, std::pair<int, double>>(std::pair<bool, std::pair<int, int>>(false, ret_rows), std::pair<int, double>(3 + num_found, ti_array[temp_ind - 1][2]));
					}
					num_found++;
				}
			}
			temp_ind = IA[++act_row];
		}
	}
	return std::pair<std::pair<bool, std::pair<int, int>>, std::pair<int, double>>(std::pair<bool, std::pair<int, int>>(true, ret_rows), std::pair<int, double>(-1, 0.0));
}

/*
Returns with
first.first: false if the check found an error,true if not
first.second.first: index of first found col
first.second.second: index of second found col
second.first: index of insertion in s_vec
second.second: new value to be inserted
*/
std::pair<std::pair<bool, std::pair<int, int>>, std::pair<int, double>> MyViewer::checkSsDown(int act_row, int act_col, int index, std::vector<double> s_vec, std::vector<double> t_vec, int viol_num, std::vector<int> excluded) {
	int i = act_col == 0 ? act_col : act_col - 1;
	int num_found = 0;
	std::pair<int, int> ret_cols = { -1, -1 };
	while (num_found < 2) {
		if (i == 0) {
			//Check whether actual column is the first one
			if (num_found == 0) ret_cols.first = i;
			else ret_cols.second = i;
			if (s_vec[1 - num_found] != si_array[0][2]) {
				//If checking for viol 1 but before that a viol2 occurs
				//In this case we should return with true(?)
				if (viol_num == 1 && s_vec[1 - num_found] > si_array[0][2])
					return std::pair<std::pair<bool, std::pair<int, int>>, std::pair<int, double>>(std::pair<bool, std::pair<int, int>>(true, ret_cols), std::pair<int, double>(-1, 0.0));
				if ((viol_num == 1 && s_vec[1 - num_found] < si_array[0][2])
					|| (viol_num == 2 && s_vec[1 - num_found] > si_array[0][2])
					|| viol_num == 0)
					return std::pair<std::pair<bool, std::pair<int, int>>, std::pair<int, double>>(std::pair<bool, std::pair<int, int>>(false, ret_cols), std::pair<int, double>(1 - num_found, si_array[0][2]));
			}
			num_found++;
		}
		else {
			std::vector<int> is_of_col = indicesOfColumn(i);
			int j = 0;
			for (;  j < is_of_col.size() && ti_array[is_of_col[j]][2] <= t_vec[2]; j++) {
			}
			//If first in act_col has bigger t than our point
			if(j==0){}
			else if (ti_array[is_of_col[j - 1]][2] < t_vec[2]) {
				//Check if not the case of last in col having smaller t than our point
				if (ti_array[is_of_col[is_of_col.size() - 1]][2] > t_vec[2]) {
					//Check whether there is an edge connecting temp_ind-1 and temp_ind,
					//meaning that a vertical ray started from our point would cut it,
					//and so the s of them should be the 1-num_found-th element of si array of our point
					bool found = false;
					int first_tmp = j - 1;
					int sec_tmp = j;
					//If one of j or j -1 is in excluded, we must check the next index accordingly
					//in order to keep the checking correct by checking only edges which were originally included
					while(first_tmp>0 && std::find(excluded.begin(), excluded.end(), is_of_col[first_tmp]) != excluded.end()) {
						first_tmp--;
					}
					while (sec_tmp < is_of_col.size()-1 && std::find(excluded.begin(), excluded.end(), is_of_col[sec_tmp]) != excluded.end()) {
						sec_tmp++;
					}
					for (int k = 0; k < edges.size() && !found; k++) {
						auto p = edges[k];
						if ((p.first == is_of_col[first_tmp]) && (p.second == is_of_col[sec_tmp])) {
							if (num_found == 0) ret_cols.first = i;
							else ret_cols.second = i;
							if (s_vec[1 - num_found] != si_array[is_of_col[first_tmp]][2]) {
								//If checking for viol 1 but before that a viol2 occurs
								//In this case we should return with true(?)
								if (viol_num == 1 && s_vec[1 - num_found] > si_array[is_of_col[first_tmp]][2])
									return std::pair<std::pair<bool, std::pair<int, int>>, std::pair<int, double>>(std::pair<bool, std::pair<int, int>>(true, ret_cols), std::pair<int, double>(-1, 0.0));
								if ((viol_num == 1 && s_vec[1 - num_found] < si_array[is_of_col[first_tmp]][2])
									|| (viol_num == 2 && s_vec[1 - num_found] > si_array[is_of_col[first_tmp]][2])
									|| viol_num == 0)
									return std::pair<std::pair<bool, std::pair<int, int>>, std::pair<int, double>>(std::pair<bool, std::pair<int, int>>(false, ret_cols), std::pair<int, double>(1 - num_found, si_array[is_of_col[first_tmp]][2]));
							}
							num_found++;
							found = true;
						}
					}
				}
			}
			//the biggest of rows with same t in act_col is earlier than ours
			else if (getRowOfExisting(is_of_col[j - 1]) < act_row) {}
			else {
				//This case occurs when ti_array[is_of_col[j-1]][2] == t_vec[2]
				while (j > 0 && act_row != getRowOfExisting(is_of_col[j - 1])) {
					j--;
				}
				//j == 0 if first in act_col has same t, but is in a row with greater index
				if (j != 0) {
					if (num_found == 0) ret_cols.first = i;
					else ret_cols.second = i;
					if (s_vec[1 - num_found] != si_array[is_of_col[j - 1]][2]) {
						//If checking for viol 1 but before that a viol2 occurs
								//In this case we should return with true(?)
						if (viol_num == 1 && s_vec[1 - num_found] > si_array[is_of_col[j - 1]][2])
							return std::pair<std::pair<bool, std::pair<int, int>>, std::pair<int, double>>(std::pair<bool, std::pair<int, int>>(true, ret_cols), std::pair<int, double>(-1, 0.0));
						if ((viol_num == 1 && s_vec[1 - num_found] < si_array[is_of_col[j - 1]][2])
							|| (viol_num == 2 && s_vec[1 - num_found] > si_array[is_of_col[j - 1]][2])
							|| viol_num == 0)
							return std::pair<std::pair<bool, std::pair<int, int>>, std::pair<int, double>>(std::pair<bool, std::pair<int, int>>(false, ret_cols), std::pair<int, double>(1 - num_found, si_array[is_of_col[j - 1]][2]));
					}
					num_found++;
				}
			}
			--i;
		}
	}
	return std::pair<std::pair<bool, std::pair<int, int>>, std::pair<int, double>>(std::pair<bool, std::pair<int, int>>(true, ret_cols), std::pair<int, double>(-1, 0.0));
}

/*
Returns with
first.first: false if the check found an error,true if not
first.second.first: index of first found col
first.second.second: index of second found col
second.first: index of insertion in s_vec
second.second: new value to be inserted
*/
std::pair<std::pair<bool, std::pair<int, int>>, std::pair<int, double>> MyViewer::checkSsUp(int act_row, int act_col, int index, std::vector<double> s_vec, std::vector<double> t_vec, int viol_num, std::vector<int> excluded) {
	int col_num = *std::max_element(JA.begin(), JA.end()) + 1;
	int i = act_col == col_num - 1 ? col_num - 1 : act_col + 1;
	int num_found = 0;
	int cpnum = tspline_control_points.size();
	std::pair<int, int> ret_cols = { -1, -1 };
	while (num_found < 2) {
		//Check whether actual column is the last one
		if (i == col_num - 1) {
			if (num_found == 0) ret_cols.first = i;
			else ret_cols.second = i;
			if (s_vec[3 + num_found] != si_array[cpnum - 1][2]) {
				//If checking for viol 1 but before that a viol2 occurs
				//In this case we should return with true(?)
				if (viol_num == 1 && s_vec[3 + num_found] < si_array[cpnum - 1][2])
					return std::pair<std::pair<bool, std::pair<int, int>>, std::pair<int, double>>(std::pair<bool, std::pair<int, int>>(true, ret_cols), std::pair<int, double>(-1, 0.0));
				//Checking any violations
				if ((viol_num == 1 && s_vec[3 + num_found] > si_array[cpnum - 1][2])
					|| (viol_num == 2 && s_vec[3 + num_found] < si_array[cpnum - 1][2])
					|| viol_num == 0)
					return std::pair<std::pair<bool, std::pair<int, int>>, std::pair<int, double>>(std::pair<bool, std::pair<int, int>>(false, ret_cols), std::pair < int, double>(3 + num_found, si_array[cpnum - 1][2]));
			}
			num_found++;
		}
		else {
			std::vector<int> is_of_col = indicesOfColumn(i);
			int j = 0;
			for (; j < is_of_col.size() && ti_array[is_of_col[j]][2] <= t_vec[2]; j++) {
			}
			//If first in act_col has bigger t than our point
			if (j == 0) {}
			else if (ti_array[is_of_col[j - 1]][2] < t_vec[2]) {
				//Check if not the case of last in col having smaller t than our point
				if (ti_array[is_of_col[is_of_col.size()-1]][2] > t_vec[2]) {
					//Check whether there is an edge connecting temp_ind-1 and temp_ind,
					//meaning that a vertical ray started from our point would cut it,
					//and so the s of them should be the 3+num_found-th element of si array of our point
					bool found = false;
					int first_tmp = j - 1;
					int sec_tmp = j;
					//If one of j or j -1 is in excluded, we must check the next index accordingly
					//in order to keep the checking correct by checking only edges which were originally included
					while (first_tmp > 0 && std::find(excluded.begin(), excluded.end(), is_of_col[first_tmp]) != excluded.end()) {
						first_tmp--;
					}
					while (sec_tmp < is_of_col.size() - 1 && std::find(excluded.begin(), excluded.end(), is_of_col[sec_tmp]) != excluded.end()) {
						sec_tmp++;
					}
					for (int k = 0; k < edges.size() && !found; k++) {
						auto p = edges[k];
						if ((p.first == is_of_col[first_tmp]) && (p.second == is_of_col[sec_tmp])) {
							if (num_found == 0) ret_cols.first = i;
							else ret_cols.second = i;
							if (s_vec[3 + num_found] != si_array[is_of_col[first_tmp]][2]) {
								//If checking for viol 1 but before that a viol2 occurs
								//In this case we should return with true(?)
								if (viol_num == 1 && s_vec[3 + num_found] < si_array[is_of_col[first_tmp]][2])
									return std::pair<std::pair<bool, std::pair<int, int>>, std::pair<int, double>>(std::pair<bool, std::pair<int, int>>(true, ret_cols), std::pair<int, double>(-1, 0.0));
								if ((viol_num == 1 && s_vec[3 + num_found] > si_array[is_of_col[first_tmp]][2])
									|| (viol_num == 2 && s_vec[3 + num_found] < si_array[is_of_col[first_tmp]][2])
									|| viol_num == 0)
									return std::pair<std::pair<bool, std::pair<int, int>>, std::pair<int, double>>(std::pair<bool, std::pair<int, int>>(false, ret_cols), std::pair<int, double>(3 + num_found, si_array[is_of_col[first_tmp]][2]));
							}
							num_found++;
							found = true;
						}
					}
				}
			}
			//the biggest of rows with same t in act_col is earlier than ours
			else if (getRowOfExisting(is_of_col[j - 1]) < act_row) {}
			else {
				//This case occurs when ti_array[is_of_col[j-1]][2] == t_vec[2]
				while (j > 0 && act_row != getRowOfExisting(is_of_col[j - 1])) {
					j--;
				}
				//j == 0 if first in act_col has same t, but is in a row with greater index
				if (j != 0) {
					if (num_found == 0) ret_cols.first = i;
					else ret_cols.second = i;
					if (s_vec[3 + num_found] != si_array[is_of_col[j - 1]][2]) {
						//If checking for viol 1 but before that a viol2 occurs
						//In this case we should return with true(?)
						if (viol_num == 1 && s_vec[3 + num_found] < si_array[is_of_col[j - 1]][2])
							return std::pair<std::pair<bool, std::pair<int, int>>, std::pair<int, double>>(std::pair<bool, std::pair<int, int>>(true, ret_cols), std::pair<int, double>(-1, 0.0));
						if ((viol_num == 1 && s_vec[3 + num_found] > si_array[is_of_col[j - 1]][2])
							|| (viol_num == 2 && s_vec[3 + num_found] < si_array[is_of_col[j - 1]][2])
							|| viol_num == 0)
							return std::pair<std::pair<bool, std::pair<int, int>>, std::pair<int, double>>(std::pair<bool, std::pair<int, int>>(false, ret_cols), std::pair<int, double>(3 + num_found, si_array[is_of_col[j - 1]][2]));
					}
					num_found++;
				}
			}
			++i;
		}
	}
	return std::pair<std::pair<bool, std::pair<int, int>>, std::pair<int, double>>(std::pair<bool, std::pair<int, int>>(true, ret_cols), std::pair<int, double>(-1, 0.0));
}

bool MyViewer::checkTSplineCorrectness() {
	int cpnum = tspline_control_points.size();
	//Check correctness of IA
	int ia_size = IA.size();
	if (IA[0] != 0) return false;
	if (IA[ia_size - 1] != cpnum) return false;
	//Check strict monotony of IA
	for (int i = 1; i < ia_size; i++) {
		if (IA[i - 1] >= IA[i]) return false;
	}
	//Check JA - whether smallest element is 0
	int min_col = *std::min_element(JA.begin(), JA.end());
	if (min_col != 0) return false;
	//Check JA - whether it contains all numbers from 0 to max_col
	int max_col = *std::max_element(JA.begin(), JA.end());
	for (int i = 0; i < max_col; i++) {
		if (std::find(JA.begin(), JA.end(), i) == JA.end()) return false;
	}
	//Check monotony of sis and tis
	for (auto si : si_array) {
		if (!std::is_sorted(si.begin(), si.end())) return false;
	}
	for (auto ti : ti_array) {
		if (!std::is_sorted(ti.begin(), ti.end())) return false;
	}

	//TODO weights must be positive??

	return true;
}

bool MyViewer::checkTSplineTopology() {
	int cpnum = tspline_control_points.size();
	std::vector<int> empty = {};
	//Check correctness of topology
	for (int i = 0; i < cpnum; i++) {
		int act_row = getRowOfExisting(i);
		int act_col = JA[i];
		if (!checkTsDown(act_row, act_col, i,si_array[i], ti_array[i], 0, empty).first.first) return false;
		if (!checkTsUp(act_row, act_col, i, si_array[i], ti_array[i], 0, empty).first.first) return false;
		if (!checkSsDown(act_row, act_col, i, si_array[i], ti_array[i], 0, empty).first.first) return false;
		if (!checkSsUp(act_row, act_col, i, si_array[i], ti_array[i], 0, empty).first.first) return false;
	}
	return true;
}

void removeEigenMColumn(MatrixXd& matrix, unsigned int colToRemove)
{
	unsigned int numRows = matrix.rows();
	unsigned int numCols = matrix.cols() - 1;

	if (colToRemove < numCols)
		matrix.block(0, colToRemove, numRows, numCols - colToRemove) = matrix.rightCols(numCols - colToRemove);

	matrix.conservativeResize(numRows, numCols);
}

//Suppose we are looking at s,t in [0,1]
void MyViewer::generatePointsAndFit() {
	std::vector<Vec> points;
	int n = 10;
	int cpnum = weights.size();

	for (size_t i = 0; i < n; ++i) {
		double t = (double)i / (double)(n - 1);
		for (size_t j = 0; j < n; ++j) {
			double s = (double)j / (double)(n - 1);
			Vec p(0.0, 0.0, 0.0);
			double nominator = 0.0;
			for (size_t k = 0; k < cpnum; ++k) {
				double B_k = cubicBSplineBasis(s, si_array[k]) * cubicBSplineBasis(t, ti_array[k]);
				p += tspline_control_points[k] * B_k;
				nominator += weights[k] * B_k;
			}
			if (abs(nominator) > 0.0) p /= nominator;
			points.push_back(p);
		}
	}
	
	fit4by4Bezier(points);
}

//S the nxn incoming points
void MyViewer::fit4by4Bezier(std::vector<Vec> S) {
	int n = sqrt(S.size());
	MatrixXd A(n*n,16), B(n*n,3);
	std::vector<Vec> P(16,Vec(0.0, 0.0, 0.0));
	std::vector<double> Bs, tempB;
	for (int i = 0; i < n; i++) {
		bernsteinAll(3, (float)i/float(n-1), tempB);
		Bs.insert(std::end(Bs), std::begin(tempB), std::end(tempB));
	}
	
	P[0] = S[0];
	P[3] = S[n-1];
	P[12] = S[(n-1)*n];
	P[15] = S[n*n-1];
	for(int t=0; t<n; t++){
		for(int s=0; s<n; s++){
			B.row(n*t + s) << S[n*t + s][0], S[n*t + s][1], S[n*t + s][2];
			for(int k=0;k<16;k++){
				A(n*t+s,k) = (Bs[s*4 + (k % 4)] * Bs[t * 4 + (k / 4)]);
			}
		}
	}

	for(int t=0; t<n; t++){
		for(int s=0; s<n; s++){
			Vec temp = A(n*t+s,0) * P[0] + A(n*t+s,3) * P[3] + A(n*t+s, 12) * P[12] + A(n*t+s, 15) * P[15];
			B(n*t + s,0) -= temp[0];
			B(n*t + s,1) -= temp[1];
			B(n*t + s,2) -= temp[2];
		}
	}

	removeEigenMColumn(A, 15);
	removeEigenMColumn(A, 12);
	removeEigenMColumn(A, 3);
	removeEigenMColumn(A, 0);

	MatrixXd X = A.colPivHouseholderQr().solve(B);
	for(int i=0;i<12;i++){
		int pInd = i + 1;
		if (i >= 2) pInd++;
		if (i >= 10) pInd++;
		P[pInd] = Vec(X(i,0), X(i, 1), X(i, 2));
	}
	
	bezier_control_points.resize(16);
	for (int i = 0; i < 16; i++) {
		bezier_control_points[i] = P[(i%4)*4 + i/4];
	}
	degree[0] = 3;
	degree[1] = 3;
	model_type = ModelType::BEZIER_SURFACE;
	/*saveBezier("fittedBezier.bzr");
	openBezier("fittedBezier.bzr");*/
}

void MyViewer::bezierToTspline() {
	tspline_control_points.resize(16);
	tspline_control_points = bezier_control_points;
	ti_array.resize(16);
	si_array.resize(16);
	JA.resize(16);
	weights.resize(16,1.0);
	blend_functions.resize(16);
	refined_points.resize(16);
	refined_weights.resize(16, { 1.0 });
	for (int i = 0; i < 16; i++) {
		tspline_control_points[i] = bezier_control_points[(i % 4) * 4 + i / 4];
		si_array[i] = i % 4 == 0 ? std::vector<double>{0,0,0,0,1} : (i%4 == 1 ? std::vector<double>{0, 0, 0, 1, 1} : (i%4==2 ? std::vector<double>{0, 0, 1, 1, 1} : std::vector<double>{ 0,1,1,1,1 }));
		ti_array[i] = i / 4 == 0 ? std::vector<double>{0, 0, 0, 0, 1} : (i / 4 == 1 ? std::vector<double>{0, 0, 0, 1, 1} : (i / 4 == 2 ? std::vector<double>{0, 0, 1, 1, 1} : std::vector<double>{ 0,1,1,1,1 }));
		JA[i] = i % 4;
		refined_points[i] = { tspline_control_points[i] };
		blend_functions[i] = { std::pair<std::vector<double>,std::vector<double>>(si_array[i],ti_array[i]) };
	}

	IA.resize(5);
	for (int i = 0; i < 5; i++) {
		IA[i] = i * 4;
	}

	updateEdgeTopology();

	std::string newFileName = fileName;
	newFileName.erase(newFileName.end()-4,newFileName.end());
	newFileName += "_bzr_to_tsp.tsp";
	model_type = ModelType::TSPLINE_SURFACE;
	/*saveTSpline(newFileName);
	openTSpline(newFileName);*/
	fileName = newFileName;
}

void MyViewer::colorDistances(std::string origFileName) {
	distColorMode = true;
	openTSpline(origFileName);
	double red = 0, green = 120, blue = 240; // Hue
	double max_dist = *std::max_element(fitDistances.begin(), fitDistances.end());
	distColors.clear();
	for (int i = 0; i < fitDistances.size(); i++) {
		double alpha = fitDistances[i]/max_dist;
		distColors.push_back(HSV2RGB({ green * (1 - alpha) + red * alpha, 1, 1 }));
	}
}

//Update M, refinement from origInd1 to origInd2 with value
void MyViewer::updateM(int origInd1, int origInd2, double value) {
	auto base = std::find(baseIndsInOrig.begin(), baseIndsInOrig.end(), origInd1);
	if (base != baseIndsInOrig.end()) {
		int baseInd = std::distance(baseIndsInOrig.begin(), base);
		//Refine from base to itself
		if (origInd1 == origInd2) {
			M(baseInd, baseInd) *= value;
		}
		//Refine from base to other
		else {
			M(baseInd, origInd2) += M(baseInd, baseInd)*value;
		}
	}
	else {
		//Refine from not base to itself
		if (origInd1 == origInd2) {
			//Multiply col with value
			M.col(origInd1) *= value;
		}
		//Refine from not base to other
		else {
			M.col(origInd2) += M.col(origInd1)*value;
		}
	}
}

void MyViewer::bring4by4ToOrig() {
	//Store original knot values
	std::string origFileName = fileName;
	origin_sarray = si_array;
	origin_tarray = ti_array;
	orig_cps = tspline_control_points;
	orig_weights = weights;

	int max_col = JA[JA.size() - 1];
	int max_row = IA.size() - 2;
	//Pushing back indices of bezier in original surface
	indsInOrig.clear();
	indsInOrig.push_back(0);
	indsInOrig.push_back(1);
	indsInOrig.push_back(IA[1] - 2);
	indsInOrig.push_back(IA[1] - 1);
	indsInOrig.push_back(IA[1]);
	indsInOrig.push_back(IA[1] + 1);
	indsInOrig.push_back(IA[2] - 2);
	indsInOrig.push_back(IA[2] - 1);
	indsInOrig.push_back(IA[max_row - 1]);
	indsInOrig.push_back(IA[max_row - 1] + 1);
	indsInOrig.push_back(IA[max_row] - 2);
	indsInOrig.push_back(IA[max_row] - 1);
	indsInOrig.push_back(IA[max_row]);
	indsInOrig.push_back(IA[max_row] + 1);
	indsInOrig.push_back(IA[max_row + 1] - 2);
	indsInOrig.push_back(IA[max_row + 1] - 1);
	baseIndsInOrig = indsInOrig;
	M = MatrixXd::Zero(16, orig_cps.size());

	JAOrig = JA;
	IAOrig = IA;
	rowsInOrig.clear();
	colsInOrig.clear();
	for (int i = 0; i < 16; i++) {
		M(i,indsInOrig[i]) = 1.0;
		rowsInOrig.emplace_back(i / 4 < 2 ? i / 4 : max_row - 1 + i / 4 - 2);
		colsInOrig.emplace_back(i % 4 < 2 ? i % 4 : max_col - 1 + i % 4 - 2);
	}
	//std::transform(indsInOrig.begin(), indsInOrig.end(), rowsInOrig.begin(), [this](int ind)->int { return getRowOfExisting(ind); });
	//std::transform(indsInOrig.begin(), indsInOrig.end(), colsInOrig.begin(), [&JA=JA](int ind)->int { return JA[ind]; });

	//fit4by4
	//Convert fitted bezier into tspline
	generatePointsAndFit();
	bezierToTspline();

	bringBackMode = true;

	//insert points until same knots->
	//calculate difference
	//color them accordingly
	bool viol = false;
	do {
		viol = false;
		for (int i = 0; i < tspline_control_points.size(); i++) {
			int act_row = getRowOfExisting(i);
			int act_col = JA[i];
			auto ss_up = checkSsUp(act_row,act_col,i,origin_sarray[i], origin_tarray[i],2,std::vector<int>());
			if (!ss_up.first.first) {
				int new_index;
				if(ss_up.second.first == 4) new_index = (getRowOfExisting(i + 1) == act_row && si_array[i + 1][2] <= origin_sarray[i][ss_up.second.first]) ? i + 2 : i + 1;
				else new_index = i + 1;
				updateOrigs(origin_sarray[i][ss_up.second.first], ti_array[i][2],new_index);
				insertRefined(origin_sarray[i][ss_up.second.first],ti_array[i][2],new_index, ss_up.second.first == 4 ? i+1 : i, ss_up.second.first == 4 ? i+2 : i+1);
				viol = true;
				break;
			}
			auto ts_up = checkTsUp(act_row, act_col, i, origin_sarray[i], origin_tarray[i], 2, std::vector<int>());
			if (!ts_up.first.first) {
				auto col_inds = indicesOfColumn(act_col);
				int indInCol = std::find(col_inds.begin(), col_inds.end(), i) - col_inds.begin();
				int new_index;
				if (ts_up.second.first == 3)
					new_index = getIndex(act_row, ts_up.first.second.first, act_col, origin_tarray[i][ts_up.second.first], false);
				else new_index = getIndex(ts_up.first.second.first, ts_up.first.second.second, act_col, origin_tarray[i][ts_up.second.first], false);
				updateOrigs(si_array[i][2], origin_tarray[i][ts_up.second.first], new_index);
				insertRefined(si_array[i][2], origin_tarray[i][ts_up.second.first], new_index, ts_up.second.first == 4 ? col_inds[indInCol + 1] : i, ts_up.second.first == 4 ? col_inds[indInCol + 2] : col_inds[indInCol + 1]);
				viol = true;
				break;
			}
		}
	} while (viol);

	saveTSpline(fileName);

	distMode = true;
	
	fitDistances.clear();
	for (int i = 0; i < tspline_control_points.size(); i++) {
		fitDistances.push_back((orig_cps[i] - tspline_control_points[i]).norm());
	}

	//colorDistances(origFileName);
	max_col = JA[JA.size() - 1];
	max_row = IA.size() - 2;
	//Pushing back indices of bezier in original surface
	indsInOrig.clear();
	indsInOrig.push_back(0);
	indsInOrig.push_back(1);
	indsInOrig.push_back(IA[1] - 2);
	indsInOrig.push_back(IA[1] - 1);
	indsInOrig.push_back(IA[1]);
	indsInOrig.push_back(IA[1] + 1);
	indsInOrig.push_back(IA[2] - 2);
	indsInOrig.push_back(IA[2] - 1);
	indsInOrig.push_back(IA[max_row - 1]);
	indsInOrig.push_back(IA[max_row - 1] + 1);
	indsInOrig.push_back(IA[max_row] - 2);
	indsInOrig.push_back(IA[max_row] - 1);
	indsInOrig.push_back(IA[max_row]);
	indsInOrig.push_back(IA[max_row] + 1);
	indsInOrig.push_back(IA[max_row + 1] - 2);
	indsInOrig.push_back(IA[max_row + 1] - 1);
	JAOrig = JA;
	IAOrig = IA;
	rowsInOrig.clear();
	colsInOrig.clear();
	for (int i = 0; i < 16; i++) {
		rowsInOrig.emplace_back(i / 4 < 2 ? i / 4 : max_row - 1 + i / 4 - 2);
		colsInOrig.emplace_back(i % 4 < 2 ? i % 4 : max_col - 1 + i % 4 - 2);
	}

	//IF coloring, need to be removed
	for (int var : indsInOrig)
	{
		fitDistances[var] = -1.0;
	}

	bezierToTspline();
	calcPointsBasedOnM();
}

//In order to update M accordingly
void MyViewer::bringToOrig() {
	distMode = false;
	bringBackMode = true;
	M = MatrixXd::Zero(baseIndsInOrig.size(),orig_cps.size());
	for (int i = 0; i<baseIndsInOrig.size(); ++i) {
		M(i, baseIndsInOrig[i]) = 1.0;
	}

	auto temp_tscps = tspline_control_points;
	auto temp_refined_points = refined_points;
	auto temp_blend_functions = blend_functions;
	auto temp_weights = weights;
	auto temp_refined_weights = refined_weights;
	auto temp_orig_rows = rowsInOrig;
	auto temp_orig_cols = colsInOrig;
	auto temp_JA = JA;
	auto temp_IA = IA;
	auto temp_siarray = si_array;
	auto temp_tiarray = ti_array;

	bool viol = false;
	do {
		viol = false;
		for (int i = 0; i < tspline_control_points.size(); i++) {
			int act_row = getRowOfExisting(i);
			int act_col = JA[i];
			auto ss_up = checkSsUp(act_row, act_col, i, origin_sarray[i], origin_tarray[i], 2, std::vector<int>());
			if (!ss_up.first.first) {
				int new_index;
				if (ss_up.second.first == 4) new_index = (getRowOfExisting(i + 1) == act_row && si_array[i + 1][2] <= origin_sarray[i][ss_up.second.first]) ? i + 2 : i + 1;
				else new_index = i + 1;
				updateOrigs(origin_sarray[i][ss_up.second.first], ti_array[i][2], new_index);
				insertRefined(origin_sarray[i][ss_up.second.first], ti_array[i][2], new_index, ss_up.second.first == 4 ? i + 1 : i, ss_up.second.first == 4 ? i + 2 : i + 1);
				viol = true;
				break;
			}
			auto ts_up = checkTsUp(act_row, act_col, i, origin_sarray[i], origin_tarray[i], 2, std::vector<int>());
			if (!ts_up.first.first) {
				auto col_inds = indicesOfColumn(act_col);
				int indInCol = std::find(col_inds.begin(), col_inds.end(), i) - col_inds.begin();
				int new_index;
				if (ts_up.second.first == 3)
					new_index = getIndex(act_row, ts_up.first.second.first, act_col, origin_tarray[i][ts_up.second.first], false);
				else new_index = getIndex(ts_up.first.second.first, ts_up.first.second.second, act_col, origin_tarray[i][ts_up.second.first], false);
				updateOrigs(si_array[i][2], origin_tarray[i][ts_up.second.first], new_index);
				insertRefined(si_array[i][2], origin_tarray[i][ts_up.second.first], new_index, ts_up.second.first == 4 ? col_inds[indInCol + 1] : i, ts_up.second.first == 4 ? col_inds[indInCol + 2] : col_inds[indInCol + 1]);
				viol = true;
				break;
			}
		}
	} while (viol);

	fitDistances.clear();
	for (int i = 0; i < tspline_control_points.size(); i++) {
		fitDistances.push_back((orig_cps[i] - tspline_control_points[i]).norm());
	}

	//IF coloring, need to be removed
	for (int var : baseIndsInOrig)
	{
		fitDistances[var] = -1.0;
	}

	tspline_control_points = temp_tscps;
	refined_points = temp_refined_points;
	blend_functions = temp_blend_functions;
	refined_weights = temp_refined_weights;
	weights = temp_weights;
	rowsInOrig = temp_orig_rows;
	colsInOrig = temp_orig_cols;
	JA = temp_JA;
	IA = temp_IA;
	si_array = temp_siarray;
	ti_array = temp_tiarray;
	indsInOrig = baseIndsInOrig;
	updateEdgeTopology(); updateMesh();
}

bool MyViewer::expandRectangleVertically(int act_row, int right_col, int left_col, int excluded) {
	int ind = IA[act_row];
	if (JA[ind] > left_col) return true;
	while (JA[ind] <= left_col) {
		ind++;
	}
	ind--;

	if (ind == excluded) return true;
	//Traverse through edges
	do
	{
		if (ind == excluded) { ++ind; }
		//More efficiently?
		if (std::find(edges.begin(), edges.end(), std::pair<int, int>(ind, ind+1 == excluded ? ind + 2 : ind + 1)) == edges.end()) {
			return true;
		}
		++ind;
	} while (JA[ind] < right_col);
	return false;
}

bool MyViewer::expandRectangleHorizontally(int act_col, int top_row, int bot_row, int excluded) {
	auto col_inds = indicesOfColumn(act_col);
	int ind = 0;
	if (getRowOfExisting(col_inds[ind]) > bot_row) return true;
	while (getRowOfExisting(col_inds[ind]) <= bot_row) {
		ind++;
	}
	ind--;

	//Traverse through edges
	do
	{
		if (col_inds[ind] != excluded) {
			if (col_inds[ind + 1] == excluded && ind + 2 >= col_inds.size()) {
				return true;
			}
			//More efficiently?
			if (std::find(edges.begin(), edges.end(), std::pair<int, int>(col_inds[ind], (col_inds[ind + 1] == excluded ? col_inds[ind + 2] : col_inds[ind + 1]))) == edges.end()) {
				return true;
			}
		}
		++ind;
	} while (ind < col_inds.size()-1 && getRowOfExisting(col_inds[ind]) < top_row);
	return false;
}

//Get the indices of the faces which the point belongs to in the case of inserting by point removal
std::pair<int,std::vector<int>> MyViewer::getFaceRectangle(int index, int act_row, int act_col, double s, double t, bool new_row, bool new_col) {
	updateEdgesTemporarily(false,index);
	ti_array.emplace(ti_array.begin() + index, std::vector<double>{-1, -1, t, -1, -1});
	si_array.emplace(si_array.begin() + index, std::vector<double>{-1,-1,s,-1,-1});
	//Find bottom row
	auto ts_down = checkTsDown(act_row, act_col, index, { 0,0,s,1,1 }, { -1,-1,t,1,1 }, 1, {});
	auto ts_up = checkTsUp(act_row, act_col, index, { 0,0,s,1,1 }, { 0,0,t,2,2 }, 1, {});
	auto ss_down = checkSsDown(act_row, act_col, index, { -1,-1,s,1,1 }, { 0,0,t,1,1 }, 1, {});
	auto ss_up = checkSsUp(act_row, act_col, index, { 0,0,s,2,2 }, { 0,0,t,1,1 }, 1, {});
	int bot_row = ts_down.first.second.first;
	int top_row = ts_up.first.second.first;
	int left_col = ss_down.first.second.first;
	int right_col = ss_up.first.second.first;

	while (expandRectangleVertically(bot_row, right_col, left_col,index)) { bot_row--; }
	while (expandRectangleVertically(top_row, right_col, left_col, index)) { top_row++; }
	while (expandRectangleHorizontally(left_col, top_row, bot_row, index)) { left_col--; }
	while (expandRectangleHorizontally(right_col, top_row, bot_row, index)) { right_col++; }

	bool sliceRow = false, sliceCol = false;
	if (new_row) {
		if (top_row > act_row) --top_row;
		if (bot_row > act_row) --bot_row;
	}
	else if (act_row < top_row && act_row > bot_row) {
		//Check whether on horizontal slicing edge of face
		if (!expandRectangleVertically(act_row, right_col, left_col,index)) {
			sliceRow = true;
		}
	}
	if (new_col) {
		if (right_col > act_col) --right_col;
		if (left_col > act_col) --left_col;
	}
	else if (act_col < right_col && act_col > left_col) {
		//Check whether on horizontal slicing edge of face
		if (!expandRectangleHorizontally(act_col, top_row, bot_row, index)) {
			sliceCol = true;
		}
	}

	updateEdgesTemporarily(true, index);
	deleteFromJA(index);
	deleteFromIA(index);
	ti_array.erase(ti_array.begin() + index);
	si_array.erase(si_array.begin() + index);
	if (sliceCol && sliceRow) {
		return { 4, {bot_row,left_col,act_row,act_col,  bot_row,act_col,act_row,right_col,  act_row,left_col,top_row,act_col,  act_row,act_col,top_row,right_col} };
	}
	else if (sliceRow) {
		return { 2, {bot_row,left_col,act_row,right_col,  act_row,left_col,top_row,right_col} };
	}
	else if (sliceCol) {
		return { 2, {bot_row,left_col,top_row,act_col,  bot_row,act_col,top_row,right_col} };
	}
	else {
		return {1,{bot_row,left_col,top_row,right_col} };
	}
}

void sortArr(std::vector<std::pair<int, int> > &vp, std::vector<int> vec)
{
	// Inserting element in pair vector 
	// to keep track of previous indexes 
	for (int i = 0; i < vec.size(); ++i) {
		vp.push_back(std::make_pair(vec[i], i));
	}

	// Sorting pair vector 
	std::stable_sort(vp.begin(), vp.end());
}

void MyViewer::calcPointsBasedOnM() {
	auto mTemp = M(all, baseIndsInOrig);
	MatrixXd origcp_m = MatrixXd(baseIndsInOrig.size(),3);
	VectorXd origw_v = VectorXd(baseIndsInOrig.size());
	for (int i = 0; i < baseIndsInOrig.size(); ++i) {
		origcp_m(i,0) = orig_cps[baseIndsInOrig[i]][0];
		origcp_m(i,1) = orig_cps[baseIndsInOrig[i]][1];
		origcp_m(i,2) = orig_cps[baseIndsInOrig[i]][2];
		origw_v(i) = orig_weights[baseIndsInOrig[i]];
	}
	mTemp.transposeInPlace();
	MatrixXd cp_sol = mTemp.colPivHouseholderQr().solve(origcp_m);
	VectorXd w_sol = mTemp.colPivHouseholderQr().solve(origw_v);
	for (int i = 0; i < baseIndsInOrig.size(); ++i) {
		tspline_control_points[i] = Vec(cp_sol(i,0), cp_sol(i, 1), cp_sol(i, 2));
		refined_points[i] = { tspline_control_points[i] };
		weights[i] = w_sol[i];
		refined_weights[i] = {weights[i]};
	}
}

void MyViewer::insertMaxDistanced() {
	distMode = true;
	bringBackMode = false;
	int max_dist_ind = std::max_element(fitDistances.begin(), fitDistances.end()) - fitDistances.begin();
	if (fitDistances[max_dist_ind] == -1.0) return;
	double new_s = origin_sarray[max_dist_ind][2];
	double new_t = origin_tarray[max_dist_ind][2];
	//Get new ind, the one from actuals which has smallest geq origind
	int new_ind = std::upper_bound(indsInOrig.begin(),indsInOrig.end(),max_dist_ind)-indsInOrig.begin();
	bool new_row = false, new_col = true;

	//Get actual row of newly inserted
	int orig_row = getRowOfExisting(max_dist_ind,true);
	int act_row;
	//If new in same orig_row as second one
	if (rowsInOrig[new_ind] == orig_row) {
		act_row = getRowOfExisting(new_ind);
		updateIA(act_row,act_row,new_t,false);
	}
	//If new in same orig_row as first one
	else if (rowsInOrig[new_ind - 1] == orig_row) {
		act_row = getRowOfExisting(new_ind - 1);
		updateIA(act_row, act_row, new_t, false);
	}
	//If between two rows
	else {
		act_row = getRowOfExisting(new_ind);
		updateIA(getRowOfExisting(new_ind - 1), act_row, new_t, false);
		new_row = true;
	}

	//Get actual col of newly inserted
	int orig_col = JAOrig[max_dist_ind];
	int act_col;
	auto existing_col = std::find(colsInOrig.begin(), colsInOrig.end(), JAOrig[max_dist_ind]);
	//If inserting in a col already existing
	if (existing_col != colsInOrig.end()) {
		act_col = JA[existing_col - colsInOrig.begin()];
		updateJA(act_col, act_col, new_ind, new_s, false);
		new_col = false;
	}
	//If not in existing col but inserting in a row
	else if (rowsInOrig[new_ind] == rowsInOrig[new_ind - 1]) {
		updateJA(JA[new_ind-1],JA[new_ind],new_ind,new_s,false);
		act_col = JA[new_ind];
	}
	//If new col and the three are not in same row
	else {
		std::vector<std::pair<int, int>> vp;
		sortArr(vp, colsInOrig);
		auto lb = std::lower_bound(vp.begin(),vp.end(),std::make_pair(orig_col,0));
		act_col = lb->second;
		updateJA(act_col,act_col+1,new_ind,new_s,false);
	}
	
	auto recEdges = getFaceRectangle(new_ind, act_row, act_col, new_s, new_t, new_row, new_col);

	//bot_row,left_col,top_row,right_col
	std::vector<int> orig_data;

	//Calculating original data before inserting anything
	for (int i = 0; i < recEdges.first; i++) {
		//Transform into orig rows and cols
		orig_data.emplace_back(rowsInOrig[IA[recEdges.second[4 * i + 0]]]);
		orig_data.emplace_back(colsInOrig[indicesOfColumn(recEdges.second[4 * i + 1])[0]]);
		orig_data.emplace_back(rowsInOrig[IA[recEdges.second[4 * i + 2]]]);
		orig_data.emplace_back(colsInOrig[indicesOfColumn(recEdges.second[4 * i + 3])[0]]);
	}
	//ISSUE when multiple faces->cols/rows need to be updated after first insertions
	for(int i=0;i<recEdges.first;i++){
		int bot_row = orig_data[4 * i];
		int left_col = orig_data[4 * i+1];
		int top_row = orig_data[4 * i+2];
		int right_col = orig_data[4 * i+3];

		//Only works correctly in cases of regular nxm grid TODO
		int vertLen = top_row - bot_row;
		int horLen = right_col - left_col;
		int botLeft = IAOrig[bot_row] + left_col;
		int topRight = IAOrig[top_row] + right_col;
		//Inserting on vertical sides
		if (vertLen > horLen) {
			//Only works correctly in cases of regular nxm grid TODO
			int ins_row = bot_row + (vertLen + 1) / 2;
			int origInd1 = IAOrig[ins_row] + left_col;
			if (fitDistances[origInd1] != -1.0) {
				int new_ind1 = std::upper_bound(indsInOrig.begin(), indsInOrig.end(), origInd1) - indsInOrig.begin();
				fitDistances[origInd1] = -1;
				//get indices of left col in orig
				auto first_col_inds = indicesOfColumn(left_col,true);
				int botLeftOrig = 0;
				//Find orig in left_col which is in actual too and just under new one
				while (getRowOfExisting(first_col_inds[botLeftOrig],true) < ins_row) { ++botLeftOrig; }
				int topLeftOrig = 1+botLeftOrig--;
				while (fitDistances[first_col_inds[botLeftOrig]] != -1) { --botLeftOrig; }
				while (fitDistances[first_col_inds[topLeftOrig]] != -1) { ++topLeftOrig; }
				//Transforming to actual
				int botLeftAct = std::distance(indsInOrig.begin(),std::find(indsInOrig.begin(), indsInOrig.end(), first_col_inds[botLeftOrig]));
				int topLeftAct = std::distance(indsInOrig.begin(),std::find(indsInOrig.begin(), indsInOrig.end(), first_col_inds[topLeftOrig]));
				indsInOrig.emplace(indsInOrig.begin() + new_ind1, origInd1);
				baseIndsInOrig.emplace(baseIndsInOrig.begin() + new_ind1, origInd1);
				rowsInOrig.emplace(rowsInOrig.begin() + new_ind1, ins_row);
				colsInOrig.emplace(colsInOrig.begin() + new_ind1, left_col);
				insertRefined(origin_sarray[origInd1][2], origin_tarray[origInd1][2], new_ind1, botLeftAct, topLeftAct);
			}

			int origInd2 = IAOrig[ins_row] + right_col;
			if (fitDistances[origInd2] != -1.0) {
				int new_ind2 = std::upper_bound(indsInOrig.begin(), indsInOrig.end(), origInd2) - indsInOrig.begin();
				fitDistances[origInd2] = -1;
				//get indices of right col in orig
				auto sec_col_inds = indicesOfColumn(right_col, true);
				int botRightOrig = 0;
				//Find orig in right_col which is in actual too and just under new one
				while (getRowOfExisting(sec_col_inds[botRightOrig], true) < ins_row) { ++botRightOrig; }
				int topRightOrig = 1 + botRightOrig--;
				while (fitDistances[sec_col_inds[botRightOrig]] != -1) { --botRightOrig; }
				while (fitDistances[sec_col_inds[topRightOrig]] != -1) { ++topRightOrig; }
				//Transforming to actual
				int botRightAct = std::distance(indsInOrig.begin(), std::find(indsInOrig.begin(), indsInOrig.end(), sec_col_inds[botRightOrig]));
				int topRightAct = std::distance(indsInOrig.begin(), std::find(indsInOrig.begin(), indsInOrig.end(), sec_col_inds[topRightOrig]));
				indsInOrig.emplace(indsInOrig.begin() + new_ind2, origInd2);
				baseIndsInOrig.emplace(baseIndsInOrig.begin() + new_ind2, origInd2);
				rowsInOrig.emplace(rowsInOrig.begin() + new_ind2, ins_row);
				colsInOrig.emplace(colsInOrig.begin() + new_ind2, right_col);
				insertRefined(origin_sarray[origInd2][2], origin_tarray[origInd2][2], new_ind2, botRightAct, topRightAct);
			}

		}
		//Inserting on horizontal sides
		else {
			//Inserting on middle of bottom face row
			int origInd1 = botLeft + (horLen + 1) / 2;
			if (fitDistances[origInd1] != -1.0) {
				int new_ind1 = std::upper_bound(indsInOrig.begin(), indsInOrig.end(), origInd1) - indsInOrig.begin();
				indsInOrig.emplace(indsInOrig.begin() + new_ind1, origInd1);
				baseIndsInOrig.emplace(baseIndsInOrig.begin() + new_ind1, origInd1);
				rowsInOrig.emplace(rowsInOrig.begin() + new_ind1, bot_row);
				colsInOrig.emplace(colsInOrig.begin() + new_ind1, left_col + (horLen + 1) / 2);
				fitDistances[origInd1] = -1;
				insertRefined(origin_sarray[origInd1][2], origin_tarray[origInd1][2], new_ind1, new_ind1 - 1, new_ind1);
			}
			//Inserting on middle of top face row
			int origInd2 = topRight - (horLen) / 2;
			if (fitDistances[origInd2] != -1.0) {
				int new_ind2 = std::upper_bound(indsInOrig.begin(), indsInOrig.end(), origInd2) - indsInOrig.begin();
				indsInOrig.emplace(indsInOrig.begin() + new_ind2, origInd2);
				baseIndsInOrig.emplace(baseIndsInOrig.begin() + new_ind2, origInd2);
				rowsInOrig.emplace(rowsInOrig.begin() + new_ind2, top_row);
				colsInOrig.emplace(colsInOrig.begin() + new_ind2, left_col + (horLen + 1) / 2);
				fitDistances[origInd2] = -1;
				insertRefined(origin_sarray[origInd2][2], origin_tarray[origInd2][2], new_ind2, new_ind2 - 1, new_ind2);
			}
		}
	}

	bringToOrig();
	calcPointsBasedOnM();
}

void MyViewer::mouseMoveEvent(QMouseEvent *e) {
  if (!axes.shown ||
      (axes.selected_axis < 0 && !(e->modifiers() & Qt::ControlModifier)) ||
      !(e->modifiers() & (Qt::ShiftModifier | Qt::ControlModifier)) ||
      !(e->buttons() & Qt::LeftButton))
    return QGLViewer::mouseMoveEvent(e);

  if (e->modifiers() & Qt::ControlModifier) {
    // move in screen plane
    double depth = camera()->projectedCoordinatesOf(axes.position)[2];
    axes.position = camera()->unprojectedCoordinatesOf(Vec(e->pos().x(), e->pos().y(), depth));
  } else {
    Vec from, dir, axis(axes.selected_axis == 0, axes.selected_axis == 1, axes.selected_axis == 2);
    camera()->convertClickToLine(e->pos(), from, dir);
    auto p = intersectLines(axes.grabbed_pos, axis, from, dir);
    float d = (p - axes.grabbed_pos) * axis;
    axes.position[axes.selected_axis] = axes.original_pos[axes.selected_axis] + d;
  }

  if (model_type == ModelType::MESH)
    mesh.set_point(MyMesh::VertexHandle(selected_vertex),
                   Vector(static_cast<double *>(axes.position)));
  if (model_type == ModelType::BEZIER_SURFACE)
    bezier_control_points[selected_vertex] = axes.position;
  if (model_type == ModelType::TSPLINE_SURFACE)
	  tspline_control_points[selected_vertex] = refined_points[selected_vertex][0] = axes.position * weights[selected_vertex];
  updateMesh();
  update();
}

void MyViewer::keyPressEvent(QKeyEvent *e) {
	if (e->modifiers() == Qt::NoModifier)
		switch (e->key()) {
		case Qt::Key_O:
			if (camera()->type() == qglviewer::Camera::PERSPECTIVE)
				camera()->setType(qglviewer::Camera::ORTHOGRAPHIC);
			else
				camera()->setType(qglviewer::Camera::PERSPECTIVE);
			update();
			break;
		case Qt::Key_P:
			visualization = Visualization::PLAIN;
			update();
			break;
		case Qt::Key_M:
			visualization = Visualization::MEAN;
			update();
			break;
		case Qt::Key_L:
			visualization = Visualization::SLICING;
			update();
			break;
		case Qt::Key_I:
			visualization = Visualization::ISOPHOTES;
			update();
			break;
		case Qt::Key_C:
			show_control_points = !show_control_points;
			update();
			break;
		case Qt::Key_S:
			show_solid = !show_solid;
			update();
			break;
		case Qt::Key_W:
			show_wireframe = !show_wireframe;
			update();
			break;
		case Qt::Key_K:
			keep_surface = !keep_surface;
			update();
			break;
			//TODO do this more user-friendly
		case Qt::Key_E:
			mid_insert = !mid_insert;
			update();
			break;
		case Qt::Key_4:
			bring4by4ToOrig();
			update();
			break;
		case Qt::Key_5:
			insertMaxDistanced();
			update();
			break;
		case Qt::Key_F:
			fairMesh();
			update();
			break;
		default:
			QGLViewer::keyPressEvent(e);
		}
	else if (e->modifiers() == Qt::KeypadModifier)
		switch (e->key()) {
		case Qt::Key_Plus:
			slicing_scaling *= 2;
			update();
			break;
		case Qt::Key_Minus:
			slicing_scaling /= 2;
			update();
			break;
		case Qt::Key_Asterisk:
			slicing_dir = Vector(static_cast<double *>(camera()->viewDirection()));
			update();
			break;
		}
	else
		QGLViewer::keyPressEvent(e);
}

QString MyViewer::helpString() const {
  QString text("<h2>Sample Framework</h2>"
               "<p>This is a minimal framework for 3D mesh manipulation, which can be "
               "extended and used as a base for various projects, for example "
               "prototypes for fairing algorithms, or even displaying/modifying "
               "parametric surfaces, etc.</p>"
               "<p>The following hotkeys are available:</p>"
               "<ul>"
               "<li>&nbsp;O: Toggle orthographic projection</li>"
               "<li>&nbsp;P: Set plain map (no coloring)</li>"
               "<li>&nbsp;M: Set mean curvature map</li>"
               "<li>&nbsp;L: Set slicing map<ul>"
               "<li>&nbsp;+: Increase slicing density</li>"
               "<li>&nbsp;-: Decrease slicing density</li>"
               "<li>&nbsp;*: Set slicing direction to view</li></ul></li>"
               "<li>&nbsp;I: Set isophote line map</li>"
               "<li>&nbsp;C: Toggle control polygon visualization</li>"
               "<li>&nbsp;S: Toggle solid (filled polygon) visualization</li>"
               "<li>&nbsp;W: Toggle wireframe visualization</li>"
               "<li>&nbsp;F: Fair mesh</li>"
               "</ul>"
               "<p>There is also a simple selection and movement interface, enabled "
               "only when the wireframe/controlnet is displayed: a mesh vertex can be selected "
               "by shift-clicking, and it can be moved by shift-dragging one of the "
               "displayed axes. Pressing ctrl enables movement in the screen plane.</p>"
               "<p>Note that libQGLViewer is furnished with a lot of useful features, "
               "such as storing/loading view positions, or saving screenshots. "
               "OpenMesh also has a nice collection of tools for mesh manipulation: "
               "decimation, subdivision, smoothing, etc. These can provide "
               "good comparisons to the methods you implement.</p>"
               "<p>This software can be used as a sample GUI base for handling "
               "parametric or procedural surfaces, as well. The power of "
               "Qt and libQGLViewer makes it easy to set up a prototype application. "
               "Feel free to modify and explore!</p>"
               "<p align=\"right\">Peter Salvi</p>");
  return text;
}
