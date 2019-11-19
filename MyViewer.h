// -*- mode: c++ -*-
#pragma once

#include <string>

#include <QGLViewer/qglviewer.h>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

using qglviewer::Vec;

class MyViewer : public QGLViewer {
  Q_OBJECT

public:
  explicit MyViewer(QWidget *parent);
  virtual ~MyViewer();

  inline double getCutoffRatio() const;
  inline void setCutoffRatio(double ratio);
  inline double getMeanMin() const;
  inline void setMeanMin(double min);
  inline double getMeanMax() const;
  inline void setMeanMax(double max);
  inline const double *getSlicingDir() const;
  inline void setSlicingDir(double x, double y, double z);
  inline double getSlicingScaling() const;
  inline void setSlicingScaling(double scaling);
  bool openMesh(const std::string &filename);
  bool openBezier(const std::string &filename);
  bool openTSpline(const std::string &filename);
  bool saveBezier(const std::string &filename);
  bool saveTSpline(const std::string &filename);

signals:
  void startComputation(QString message);
  void midComputation(int percent);
  void endComputation();

protected:
  virtual void init() override;
  virtual void draw() override;
  virtual void drawWithNames() override;
  virtual void endSelection(const QPoint & p) override;
  virtual void postSelection(const QPoint &p) override;
  virtual void keyPressEvent(QKeyEvent *e) override;
  virtual void mouseMoveEvent(QMouseEvent *e) override;
  virtual QString helpString() const override;

private:
  struct MyTraits : public OpenMesh::DefaultTraits {
    using Point  = OpenMesh::Vec3d; // the default would be Vec3f
    using Normal = OpenMesh::Vec3d;
    VertexTraits {
      double mean;              // approximated mean curvature
    };
  };
  using MyMesh = OpenMesh::TriMesh_ArrayKernelT<MyTraits>;
  using Vector = OpenMesh::VectorT<double,3>;

  // Mesh
  void updateMesh(bool update_mean_range = true);
  void updateVertexNormals();
  void localSystem(const Vector &normal, Vector &u, Vector &v);
  double voronoiWeight(MyMesh::HalfedgeHandle in_he);
  void updateMeanMinMax();
  void updateMeanCurvature(bool update_min_max = true);

  // Bezier
  static void bernsteinAll(size_t n, double u, std::vector<double> &coeff);
  void generateBezierMesh();

  //TSpline
  double cubicBSplineBasis(double param, std::vector<double> knots);
  void generateTSplineMesh();
  void updateEdgeTopology();
  bool checkTSplineCorrectness();
  bool checkTSplineTopology();
  std::vector<int> indicesOfColumn(int colindex);
  int actRow(int index);
  std::pair<bool, std::pair<int, double>> checkTsDown(int index, int viol_num);
  std::pair<bool, std::pair<int, double>> checkTsUp(int index, int viol_num);
  std::pair<bool, std::pair<int, double>> checkSsDown(int index, int viol_num);
  std::pair<bool, std::pair<int, double>> checkSsUp(int index, int viol_num);
  std::pair<std::vector<int>, std::vector<double>> refineRowCol(double new_value, int row_col_ind, bool is_row);
  void checkViolation(std::vector<int> indices, int new_ind, bool is_row, bool is_first);
  std::pair<double, double> refineBlend(std::vector<double> knot_vec, int ins_ind, double new_value);
  bool checkForViol1();
  bool checkForViol2();
  void checkViolations();
  std::pair<bool, int> getIndex(double s, double t);
  std::pair<bool, int> getRow(double t);

  // Visualization
  void setupCamera();
  Vec meanMapColor(double d) const;
  void drawBezierControlNet() const;
  void drawTSplineControlNet(bool with_names, int start_index) const;
  void drawAxes() const;
  void drawAxesWithNames() const;
  static Vec intersectLines(const Vec &ap, const Vec &ad, const Vec &bp, const Vec &bd);

  // Other
  void fairMesh();

  //////////////////////
  // Member variables //
  //////////////////////

  enum class ModelType { NONE, MESH, BEZIER_SURFACE, TSPLINE_SURFACE } model_type;

  // Mesh
  MyMesh mesh;

  // Bezier
  size_t degree[2];
  std::vector<Vec> bezier_control_points;

  //TSpline
  //https://www.geeksforgeeks.org/sparse-matrix-representations-set-3-csr/ - sparse matrix representation

  std::vector<Vec> tspline_control_points;
  std::vector<int> IA;
  std::vector<int> JA;

  std::vector<std::vector<double>> si_array;
  std::vector<std::vector<double>> ti_array;
  std::vector<double> weights;
  //In order to handle edges
  std::vector<std::pair<int,int>> edges;
  std::vector<std::pair<double,double>> blend_multipliers;

  // Visualization
  double mean_min, mean_max, cutoff_ratio;
  bool show_control_points, show_solid, show_wireframe;
  enum class Visualization { PLAIN, MEAN, SLICING, ISOPHOTES } visualization;
  GLuint isophote_texture, slicing_texture;
  Vector slicing_dir;
  double slicing_scaling;
  int selected_vertex;
  struct ModificationAxes {
    bool shown;
    float size;
    int selected_axis;
    Vec position, grabbed_pos, original_pos;
  } axes;
};

#include "MyViewer.hpp"
