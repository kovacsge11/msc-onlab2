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
  void updateEdgesTemporarily(bool remove, int temp_index);
  bool checkTSplineCorrectness();
  bool checkTSplineTopology();
  std::vector<int> indicesOfColumn(int colindex);
  std::pair<std::pair<bool, int>, std::pair<int, double>> checkTsDown(int act_row, int act_col, int index, std::vector<double> s_vec, std::vector<double> t_vec, int viol_num);
  std::pair<std::pair<bool, int>, std::pair<int, double>> checkTsUp(int act_row, int act_col, int index, std::vector<double> s_vec, std::vector<double> t_vec, int viol_num);
  std::pair<std::pair<bool, int>, std::pair<int, double>> checkSsDown(int act_row, int act_col, int index, std::vector<double> s_vec, std::vector<double> t_vec, int viol_num);
  std::pair<std::pair<bool, int>, std::pair<int, double>> checkSsUp(int act_row, int act_col, int index, std::vector<double> s_vec, std::vector<double> t_vec, int viol_num);
  std::pair<std::vector<int>, std::vector<double>> refineRowCol(double new_value, int row_col_ind, bool is_row);
  std::pair<std::pair<double, std::vector<double>>, std::pair<double, std::vector<double>>> refineBlend(std::vector<double> knot_vec, int ins_ind, double new_value);
  bool checkForViol1(std::vector<int> excluded);
  std::pair<bool, std::vector<int>> checkForViol2(std::vector<int> excluded);
  void checkViolations(std::vector<int> excluded);
  int getIndex(int first_row, int sec_row, int act_col, double t);
  std::pair<bool, int> getRowOfNew(int first_row, int sec_row, double t);
  int MyViewer::getRowOfExisting(int index);
  std::pair<bool, int> getColOfNew(int first_col, int sec_col, double s);
  void updateIA(int first_row, int sec_col, double t);
  void updateJA(int first_col, int sec_col, int new_ind, double s);
  void deleteFromIA(int del_ind);
  void deleteFromJA(int del_ind);
  bool edgeExists(int first_ind, int sec_ind);
  void insertRefined(double s, double t, int new_ind, int first_ind, int sec_ind);
  void insertOnFace(int first_ind, int sec_ind);
  std::pair<bool, double> checkOpposite(int act_row, int act_col, double s, double t, bool horizontal_insertion, int new_index, double epsilon);

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
  std::vector<std::vector<std::pair<std::vector<double>, std::vector<double>>>> blend_functions;
  //For every blend function of every point stores the coordinates of actual point of origin multiplied by the factor of refinements
  std::vector<std::vector<Vec>> refined_points;
  //For every blend function of every point stores the weight of actual point of origin multiplied by the factor of refinements
  std::vector<std::vector<double>> refined_weights;

  bool keep_surface, mid_insert;

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
