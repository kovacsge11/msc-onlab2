#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

#include <QtGui/QKeyEvent>

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Tools/Smoother/JacobiLaplaceSmootherT.hh>

// #define BETTER_MEAN_CURVATURE

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
  mean_min(0.0), mean_max(0.0), cutoff_ratio(0.05),
  show_control_points(true), show_solid(true), show_wireframe(false),
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
		for (size_t i = 0; i < cpnum; ++i){
			si_array[i].resize(5);
			ti_array[i].resize(5);
			f >> tspline_control_points[i][0] >> tspline_control_points[i][1] >> tspline_control_points[i][2];
			f >> si_array[i][0] >> si_array[i][1] >> si_array[i][2] >> si_array[i][3] >> si_array[i][4];
			f >> ti_array[i][0] >> ti_array[i][1] >> ti_array[i][2] >> ti_array[i][3] >> ti_array[i][4];
			f >> weights[i];
			//Filling up JA vector as well
			f >> JA[i];
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
	size_t cpnum = tspline_control_points.size();
	size_t ia_size = IA.size();
	//Storing controlnet in row direction
	for (size_t i = 1; i < ia_size; ++i) {
		bool first = true;
		for (size_t j = IA[i - 1]; j < IA[i]; ++j) {
			//If last in row or is not connected with next in row /this way the topology is surely true to rule 2/
			if ((si_array[j][3] == si_array[j][2]) || (si_array[j][3] != si_array[j + 1][2])) {
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
	for (size_t i = 0; i < col_num; ++i) {
		first = true;
		std::vector<int> col_indices = indicesOfColumn(i);
		for (size_t j = 0; j < col_indices.size(); ++j) {
			//If last or first in column or is not connected with next in column /this way the topology is surely true to rule 2/
			if (j==col_indices.size()-1 || j==0 || (ti_array[col_indices[j]][2] != ti_array[col_indices[j-1]][3])) {
				if (!first) {
					first = true;
					std::pair<int, int> ind_pair = std::pair<int, int>(col_indices[j - 1], col_indices[j]);
					edges.push_back(ind_pair);
				}
				else { first = false; }
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
		const auto &p1 = tspline_control_points[ind_pair.first];
		const auto &p2 = tspline_control_points[ind_pair.second];
		glVertex3dv(p1);
		glVertex3dv(p2);
		glEnd();
		if(with_names) glPopName();
	}

	//Drawing points
	glLineWidth(1.0);
	glPointSize(8.0);
	glColor3d(1.0, 0.0, 1.0);
	glBegin(GL_POINTS);
	for (const auto &pn : tspline_control_points)
		glVertex3dv(pn);
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
		  Vec const &p = tspline_control_points[i];
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
			for (int i = 0; i < nbHits; ++i)
				//If a point is selected, too
				if ((selectBuffer())[i * 4 + 3] < tspline_control_points.size()) {
					setSelectedName((selectBuffer())[i * 4 + 3]);
					return;
				}
			setSelectedName((selectBuffer())[3]);
		}
	}
}

std::vector<int> MyViewer::indicesOfColumn(int colindex) {
	std::vector<int> ret_vec;
	ret_vec.clear();
	for (int i = 0; i < JA.size(); i++) {
		if (JA[i] == colindex) ret_vec.push_back(i);
	}
	return ret_vec;
}

void MyViewer::postSelection(const QPoint &p) {
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
	  else axes.position = tspline_control_points[sel];
  if (!edge) {
	  double depth = camera()->projectedCoordinatesOf(axes.position)[2];
	  Vec q1 = camera()->unprojectedCoordinatesOf(Vec(0.0, 0.0, depth));
	  Vec q2 = camera()->unprojectedCoordinatesOf(Vec(width(), height(), depth));
	  axes.size = (q1 - q2).norm() / 10.0;
	  axes.shown = true;
	  axes.selected_axis = -1;
  }
  else {
	  bool found;
	  //Vec selectedPoint = camera()->pointUnderPixel(p, found);
	  std::pair<int, int> index_pair = edges[sel - cpnum];
	  Vec selectedPoint = (tspline_control_points[index_pair.first] + tspline_control_points[index_pair.second]) / 2.0;
	  std::vector<double> new_si, new_ti;
	  int new_index;
	  double new_s, new_t;
	  //If in same row, otherwise they must be in same column
	  if (ti_array[index_pair.first][2] == ti_array[index_pair.second][2]) {
		  new_s = (si_array[index_pair.first][2] + si_array[index_pair.second][2]) / 2.0;
		  new_si = { si_array[index_pair.first][1], si_array[index_pair.first][2], new_s, si_array[index_pair.second][2], si_array[index_pair.second][3]};
		  new_t = ti_array[index_pair.first][2];
		  new_index = index_pair.second;

		  //Finding new ti
		  new_ti.clear();
		  
		  int act_row = actRow(index_pair.first);
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
				  for (; si_array[temp_ind][2] <= new_s && actRow(temp_ind) == i; temp_ind++) {
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
				  else if (i != actRow(temp_ind - 1)) {}
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
				  for (; si_array[temp_ind][2] <= new_s && actRow(temp_ind) == i; temp_ind++) {
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
				  else if (i != actRow(temp_ind - 1)) {}
				  else {
					  //This case occurs when si_array[temp_ind - 1][2] == new_s
					  new_ti.push_back(ti_array[temp_ind - 1][2]);
					  num_found++;
				  }
				  temp_ind = IA[++i];
			  }
		  }

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
		  //Update JA too
		  int lower_col = JA[index_pair.first];
		  int upper_col = JA[index_pair.second];
		  bool found = false;
		  while(!found) {
			  //If low_col in same col as new point -- floating point comparison
			  if (si_array[indicesOfColumn(lower_col)[0]][2] == new_s) {
				  JA.insert(JA.begin() + new_index, lower_col);
				  found = true;
			  }
			  //If new col must be inserted in JA
			  else{
				  if (lower_col + 1 == upper_col) {
					  for (int j = 0; j < JA.size(); j++) {
						  if (JA[j] > lower_col) JA[j] += 1;
					  }
					  JA.insert(JA.begin() + new_index, lower_col + 1);
					  found = true;
				  }
				  else { lower_col++; }
			  }
		  }
	  }
	  else {
		  new_s = si_array[index_pair.first][2];
		  new_t = (ti_array[index_pair.first][2] + ti_array[index_pair.second][2]) / 2.0;
		  new_ti = { ti_array[index_pair.first][1], ti_array[index_pair.first][2], new_t, ti_array[index_pair.second][2], ti_array[index_pair.second][3] };
		  
		  //Finding new index
		  int i = 0;
		  for (; IA[i] <= index_pair.first; i++) {
		  }
		  int temp_ind = IA[i]; //start index of row (of first)+1
		  while (i < IA.size()-1) {
			  if (ti_array[temp_ind][2] > new_t) {
				  new_index = temp_ind;
				  i = IA.size(); //to finish iterating
			  }
			  else if (ti_array[temp_ind][2] < new_t) temp_ind = IA[++i]; //go to next row
			  else {
				  //iterate through this row
				  for (; temp_ind < IA[i + 1] && si_array[temp_ind][2] < new_s; temp_ind++) {
				  }
				  if (temp_ind == IA[i + 1] || si_array[temp_ind][2] > new_s) {
					  new_index = temp_ind;
					  i = IA.size(); //to finish iterating
				  }
				  else return; //point already exists
			  }
		  }
		  //If single point between last and second to last row
		  if (i == IA.size() - 1) new_index = temp_ind;

		  //Finding new si
		  new_si.clear();
		  int act_col;
		  int col_num = *std::max_element(JA.begin(), JA.end()) + 1;
		  act_col = JA[index_pair.first];

		  //Check s-s downwards
		  i = act_col==0 ? act_col : act_col-1;
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

		  //Updating IA and JA matrices too
		  //Update JA
		  JA.insert(JA.begin()+new_index,act_col);
		  //Update IA too
		  int lower_row = 0;
		  for (; IA[lower_row] <= index_pair.first; lower_row++) {
		  }
		  lower_row--;
		  int upper_row = lower_row;
		  for (; IA[upper_row] <= index_pair.second; upper_row++) {
		  }
		  upper_row--;
		  found = false;
		  while (!found && lower_row <= upper_row) {
			  //If low_col in same col as new point -- floating point comparison
			  if (ti_array[IA[lower_row]][2] == new_t) {
				  for (int j = lower_row + 1; j < IA.size(); j++) {
					  IA[j] += 1;
				  }
				  found = true;
			  }
			  //If new row must be inserted in IA
			  else if (ti_array[IA[lower_row]][2] > new_t) {
				  for (int j = lower_row; j < IA.size(); j++) {
					  IA[j] += 1;
				  }
				  IA.insert(IA.begin() + lower_row, IA[lower_row] - 1);
				  found = true;
			  }
			  else {
				  lower_row++;
			  }
		  }
	  }
	  double weight = 1.0; //TODO other weight value??
	  weights.insert(weights.begin() + new_index, weight);
	  si_array.insert(si_array.begin() + new_index, new_si);
	  ti_array.insert(ti_array.begin() + new_index, new_ti);
	  tspline_control_points.insert(tspline_control_points.begin() + new_index, selectedPoint);
	  updateEdgeTopology();
  }
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
    } else
    QGLViewer::keyPressEvent(e);
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

double MyViewer::cubicBSplineBasis(bool is_s, double param, int cpt_indx) {
	const auto &knots = is_s ? si_array[cpt_indx] : ti_array[cpt_indx];
	double u = param;
	size_t p = 3, i;
	if (u < knots.front() || u > knots.back())
		return 0.0;
	if (u == knots.back())
		i = knots.size() - 1;
	else
		i = (std::upper_bound(knots.begin(), knots.end(), u) - knots.begin()) - 1;
	std::vector<double> coeff; coeff.resize(p + 1, 0.0);
	coeff[i] = 1.0;
	for (size_t j = 1; j <= p; ++j)
		for (size_t k = 0; k <= p - j; ++k)
			coeff[k] =
			(coeff[k] ? coeff[k] * (u - knots[k]) / (knots[k + j] - knots[k]) : 0.0) +
			(coeff[k + 1] ? coeff[k + 1] * (knots[k + j + 1] - u) / (knots[k + j + 1] - knots[k + 1]) : 0.0);
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
	size_t resolution = 30;
	size_t cpnum = weights.size();
	//Assuming that the last point is the one with both the biggest s and biggest t -->cause surface cpts: rectangle
	double biggest_s = si_array[cpnum-1][2];
	double biggest_t = ti_array[cpnum-1][2];
	//Assuming that the first point is the one with both the smallest s and smallest t
	double smallest_s = si_array[0][2];
	double smallest_t = ti_array[0][2];
	double s_range = biggest_s - smallest_s;
	double t_range = biggest_t - smallest_t;
	//A small offset, cause point generation doesn't work on edge of topology(true?)
	double s_epsilon = s_range / (double)((resolution-1)*(resolution - 1));
	double t_epsilon = t_range / (double)((resolution - 1)*(resolution - 1));
	smallest_s += s_epsilon;
	smallest_t += t_epsilon;
	biggest_s -= s_epsilon;
	biggest_t -= t_epsilon;
	s_range -= 2 * s_epsilon;
	t_range -= 2 * t_epsilon;

	mesh.clear();
	std::vector<MyMesh::VertexHandle> handles, tri;

	std::vector<double> coeff_s, coeff_t;
	for (size_t i = 0; i < resolution; ++i) {
		double s = smallest_s + (s_range * (double)i / (double)(resolution - 1));
		for (size_t j = 0; j < resolution; ++j) {
			double t = smallest_t + (t_range * (double)j / (double)(resolution - 1));
			Vec p(0.0, 0.0, 0.0);
			double nominator = 0.0;
			for (size_t k = 0; k < cpnum; ++k) {
				double B_k = cubicBSplineBasis(true,s,k) * cubicBSplineBasis(false,t,k);
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

int MyViewer::actRow(int index) {
	int act_row = 0;
	for (; IA[act_row] <= index; act_row++) {
	}
	act_row--;
	return act_row;
}

//TODO moving point after multiple insertion made an error in surface - right side disappeared

//TODO organize postSelection blocks into these 4 checkfunctions
//TODO correct elseif-else branches there too
bool MyViewer::checkTsDown(int index) {
	int act_row = actRow(index);
	int temp_ind = act_row == 0 ? 0 : IA[--act_row]; //start index of row(of index)+1
	int num_found = 0;
	while (num_found < 2) {
		//Check whether actual row is the first one
		if (temp_ind == 0) {
			if (ti_array[index][1-num_found] != ti_array[0][2]) return false;
			num_found++;
		}
		else {
			for (; si_array[temp_ind][2] <= si_array[index][2] && actRow(temp_ind) == act_row; temp_ind++) {
			}
			if (si_array[temp_ind - 1][2] < si_array[index][2]) {
				//Check if not the case of last in row having smaller s than the point with index "index"
				if (si_array[temp_ind][2] > si_array[index][2]) {
					//check whether there is an edge connecting temp_ind-1 and temp_ind,
					//meaning that a vertical ray started from our point would cut it,
					//and so the t of them should be the 1-num_found-th element of ti array of our point
					bool found = false;
					for (int j = 0; j < edges.size(), !found; j++) {
						auto p = edges[j];
						if ((p.first == temp_ind - 1) && (p.second == temp_ind)) {
							if (ti_array[index][1 - num_found] != ti_array[temp_ind - 1][2]) return false;
							num_found++;
							found = true;
						}
					}
				}
			} //First of actual row has greater s than our point
			else if (act_row != actRow(temp_ind - 1)) {}
			else {
				//This case occurs when si_array[temp_ind - 1][2] == si_array[index][2]
				if (ti_array[index][1-num_found] != ti_array[temp_ind - 1][2]) return false;
				num_found++;
			}
			temp_ind = IA[--act_row];
		}
	}
	return true;
}

bool MyViewer::checkTsUp(int index) {
	int act_row = actRow(index);
	int temp_ind = act_row == IA.size() - 2 ? IA[IA.size() - 2] : IA[++act_row]; //start index of row(of index)+1
	int num_found = 0;
	int cpnum = tspline_control_points.size();
	while (num_found < 2) {
		//Check whether actual row is the last one
		if (temp_ind == IA[IA.size() - 2]) {
			if (ti_array[index][3+num_found] != ti_array[cpnum-1][2]) return false;
			num_found++;
		}
		else {
			for (; si_array[temp_ind][2] <= si_array[index][2] && actRow(temp_ind) == act_row; temp_ind++) {
			}
			if (si_array[temp_ind - 1][2] < si_array[index][2]) {
				//Check if not the case of last in row having smaller s than our point
				if (si_array[temp_ind][2] > si_array[index][2]) {
					//Check whether there is an edge connecting temp_ind-1 and temp_ind,
					//meaning that a vertical ray started from our point would cut it,
					//and so the t of them should be the 3+num_found-th element of ti array of our point
					bool found = false;
					for (int j = 0; j < edges.size(), !found; j++) {
						auto p = edges[j];
						if ((p.first == temp_ind - 1) && (p.second == temp_ind)) {
							if (ti_array[index][3 + num_found] != ti_array[temp_ind - 1][2]) return false;
							num_found++;
							found = true;
						}
					}
				}
			} //First of actual row has greater s than our point
			else if(act_row != actRow(temp_ind-1)){}
			else {
				//This case occurs when si_array[temp_ind - 1][2] == si_array[index][2]
				if (ti_array[index][3 + num_found] != ti_array[temp_ind - 1][2]) return false;
				num_found++;
			}
			temp_ind = IA[++act_row];
		}
	}
	return true;
}

bool MyViewer::checkSsDown(int index) {
	int act_col = JA[index];
	int i = act_col == 0 ? act_col : act_col - 1;
	int num_found = 0;
	while (num_found < 2) {
		if (i == 0) {
			//Check whether actual column is the first one
			if (si_array[index][1-num_found] != si_array[0][2]) return false;
			num_found++;
		}
		else {
			std::vector<int> is_of_col = indicesOfColumn(i);
			int j = 0;
			for (;  j < is_of_col.size() && ti_array[is_of_col[j]][2] <= ti_array[index][2]; j++) {
			}
			//If first in act_col has bigger t than our point
			if(j==0){}
			else if (ti_array[is_of_col[j - 1]][2] < ti_array[index][2]) {
				//Check if not the case of last in col having smaller t than our point
				if (ti_array[is_of_col[j]][2] > ti_array[index][2]) {
					//Check whether there is an edge connecting temp_ind-1 and temp_ind,
					//meaning that a vertical ray started from our point would cut it,
					//and so the s of them should be the 1-num_found-th element of si array of our point
					bool found = false;
					for (int k = 0; k < edges.size(), !found; k++) {
						auto p = edges[k];
						if ((p.first == is_of_col[j - 1]) && (p.second == is_of_col[j])) {
							if (si_array[index][1 - num_found] != si_array[is_of_col[j - 1]][2]) return false;
							num_found++;
							found = true;
						}
					}
				}
			}
			else {
				//This case occurs when ti_array[is_of_col[j-1]][2] == ti_array[index][2]
				if(si_array[index][1-num_found] != si_array[is_of_col[j - 1]][2]) return false;
				num_found++;
			}
			--i;
		}
	}
	return true;
}

bool MyViewer::checkSsUp(int index) {
	int act_col = JA[index];
	int col_num = *std::max_element(JA.begin(), JA.end()) + 1;
	int i = act_col == col_num - 1 ? col_num - 1 : act_col + 1;
	int num_found = 0;
	int cpnum = tspline_control_points.size();
	while (num_found < 2) {
		//Check whether actual column is the last one
		if (i == col_num - 1) {
			if (si_array[index][3 + num_found] != si_array[cpnum - 1][2]) return false;
			num_found++;
		}
		else {
			std::vector<int> is_of_col = indicesOfColumn(i);
			int j = 0;
			for (; j < is_of_col.size() && ti_array[is_of_col[j]][2] <= ti_array[index][2]; j++) {
			}
			//If first in act_col has bigger t than our point
			if (j == 0) {}
			else if (ti_array[is_of_col[j - 1]][2] < ti_array[index][2]) {
				//Check if not the case of last in col having smaller t than our point
				if (ti_array[is_of_col[j]][2] > ti_array[index][2]) {
					//Check whether there is an edge connecting temp_ind-1 and temp_ind,
					//meaning that a vertical ray started from our point would cut it,
					//and so the s of them should be the 3+num_found-th element of si array of our point
					bool found = false;
					for (int k = 0; k < edges.size(), !found; k++) {
						auto p = edges[k];
						if ((p.first == is_of_col[j - 1]) && (p.second == is_of_col[j])) {
							if (si_array[index][3 + num_found] != si_array[is_of_col[j - 1]][2]) return false;
							num_found++;
							found = true;
						}
					}
				}
			}
			else {
				//This case occurs when ti_array[is_of_col[j-1]][2] == ti_array[index][2]
				if (si_array[index][3+num_found] != si_array[is_of_col[j - 1]][2]) return false;
				num_found++;
			}
			++i;
		}
	}
	return true;
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
	//Check correctness of topology
	for (int i = 0; i < cpnum; i++) {
		if (!checkTsDown(i)) return false;
		if (!checkTsUp(i)) return false;
		if (!checkSsDown(i)) return false;
		if (!checkSsUp(i)) return false;
	}
	return true;
}

/*void MyViewer::blendFuncRefine(double s, double t) {
	//for()
}*/

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
	  tspline_control_points[selected_vertex] = axes.position;
  updateMesh();
  update();
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
