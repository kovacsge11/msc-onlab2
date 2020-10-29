// -*- mode: c++ -*-
#pragma once

#include <string>

#include <QGLViewer/qglviewer.h>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include "Eigen/Core"

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
  void bSplineBasis(double u, const std::vector<double>& knots, int degree,
	  std::vector<double>& coeff);
  void generateTSplineMesh();
  void updateEdgeTopology();
  void updateEdgesTemporarily(bool remove, int temp_index);
  bool checkTSplineCorrectness();
  bool checkTSplineTopology();
  std::pair<std::pair<bool, std::pair<int, int>>, std::pair<int, double>> checkTsDown(int act_row, int act_col, std::vector<double>& s_vec, std::vector<double>& t_vec, int viol_num, const std::vector<int>& excluded);
  std::pair<std::pair<bool, std::pair<int, int>>, std::pair<int, double>> checkTsUp(int act_row, int act_col, std::vector<double>& s_vec, std::vector<double>& t_vec, int viol_num, const std::vector<int>& excluded);
  std::pair<std::pair<bool, std::pair<int, int>>, std::pair<int, double>> checkSsDown(int act_row, int act_col, std::vector<double>& s_vec, std::vector<double>& t_vec, int viol_num, const std::vector<int>& excluded);
  std::pair<std::pair<bool, std::pair<int, int>>, std::pair<int, double>> checkSsUp(int act_row, int act_col, std::vector<double>& s_vec, std::vector<double>& t_vec, int viol_num, const std::vector<int>& excluded);
  std::pair<std::vector<int>, std::vector<double>> refineRowCol(double new_value, int row_col_ind, bool is_row);
  std::pair<std::pair<double, std::vector<double>>, std::pair<double, std::vector<double>>> refineBlend(std::vector<double>& knot_vec, int ins_ind, double new_value);
  std::pair<bool, std::pair<std::vector<int>, std::vector<int>>> checkForViol1(std::vector<int>& excluded, std::vector<int>& newlyAdded);
  std::pair<bool, std::pair<std::vector<int>, std::vector<int>>> checkForViol2(std::vector<int>& excluded, std::vector<int>& newlyAdded);
  void updateOrigs(double s, double t, int act_ind, int orig_min_row, int orig_min_col, bool use_min_col_as_exact);
  std::pair<std::vector<int>, std::vector<int>> insertAfterViol(int new_index, std::vector<double>& new_si, std::vector<double>& new_ti, std::vector<int>& excluded, std::vector<int>& newlyAdded);
  void checkViolations(std::vector<int>& excluded);
  std::vector<int> indicesOfColumn(int colindex, bool inOrig = false);
  int getIndexWhenColInsert(int first_row, int sec_row, int act_col, double t, bool maxFromEquals);
  int getIndexWhenRowInsert(int row, int act_col);
  std::pair<bool, int> getRowOfNew(int first_row, int sec_row, double t, bool maxFromEquals, bool use_orig_ind, int new_ind_to_be);
  int getRowOfExisting(int index, bool inOrig = false);
  std::pair<bool, int> getColOfNew(int first_col, int sec_col, double s, bool maxFromEquals, bool use_orig_ind, int new_ind_to_be);
  void updateIA(int first_row, int sec_col, double t, bool maxFromEquals, int new_ind, bool use_orig_ind);
  void updateJA(int first_col, int sec_col, int new_ind, double s, bool maxFromEquals, bool use_orig_ind);
  void deleteFromIA(int del_ind);
  void deleteFromJA(int del_ind);
  bool edgeExists(int first_ind, int sec_ind);
  void insertRefined(double s, double t, int new_ind, int first_row, int sec_row, int first_col,
	  int sec_col, bool use_orig_inds_for_JA);
  std::pair<bool, double> checkOpposite(int act_row, int act_col, double s, double t, bool horizontal_insertion, int new_index, double epsilon);
  void fit4by4Bezier(const std::vector<Vec>& S, const std::vector<double>& us,
	  const std::vector<double>& vs, const std::vector<int>& corner_inds);
  void fitTSpline(const std::vector<Vec>& S, const std::vector<double>& sample_points_us,
	  const std::vector<double>& sample_points_vs, const std::vector<int>& sample_corner_inds,
	  const std::vector<std::vector<double>>& param_si_array,
	  const std::vector<std::vector<double>>& param_ti_array, const std::vector<int>& fit_corner_inds);
  void fitSpline(const std::vector<Vec>& S, const std::vector<double>& sample_points_us,
	  const std::vector<double>& sample_points_vs, const std::vector<int>& sample_corner_inds,
	  const std::vector<std::vector<double>>& param_si_array,
	  const std::vector<std::vector<double>>& param_ti_array, const std::vector<int>& fit_corner_inds,
	  std::vector<Vec>& return_pts);
  void readObjWithUV();
  void exampleFit();
  void fitPointCloudIter();
  void fitPointCloud(const std::vector<Vec>& sample_points, std::vector<double>& us,
	  std::vector<double>& vs, const std::vector<int>& sample_corner_inds);
  void generatePoints(std::vector<Vec>& points, int n,
	  std::vector<double>& return_us, std::vector<double>& return_vs,
	  std::vector<int>& return_corner_inds);
  void bezierToTspline();
  void newtonRaphsonProjection(double& u, double& v, const Vec& p, int max_iter,
	  double dist_tol, double cos_tol);
  void evaluateTSpline(double u, double v, Vec& return_p);
  void calculateDer(double u, double v, int grade_u, int grade_v, Vec& result);
  void checkColsBetween(int first_col, int last_col, int row, std::vector<double>& result,
	  double s, double t);
  void knotVectorInRow(int row, std::vector<double>& result);
  void checkRowsBetween(int first_row, int last_row, int col, std::vector<double>& result,
	  double s, double t);
  void knotVectorInCol(int col, std::vector<double>& result);
  void colorDistances(std::string origFileName);
  void updateM(int origInd1, int origInd2, double value, std::vector<int>& origIndRefineOrig);
  void bringBackIterations();
  void bring4by4ToOrig();
  void bringToOrig();
  void calcPointsBasedOnM();
  void findRowBasedOnKnot(double v, int& ret_row, bool& new_row);
  void findColBasedOnKnot(double u, int& ret_col, bool& new_col);
  bool checkWhetherPointExists(int row, int col);
  void insertMaxDistancedWithoutOrig(double u, double v, std::vector<int>& corner_inds);
  void insertMaxDistancedWithOrig();
  std::pair<int, std::vector<int>> getFaceRectangle(int index, int act_row, int act_col, double s, double t, bool new_row, bool new_col);
  bool expandRectangleVertically(int act_row, int right_col, int left_col, int excluded);
  bool expandRectangleHorizontally(int act_col, int top_row, int bot_row, int excluded);

  // Visualization
  void setupCamera();
  Vec meanMapColor(double d) const;
  void drawBezierControlNet() const;
  void drawTSplineControlNet(bool with_names, int start_index) const;
  void drawPointClouds() const;
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
  std::string fileName;

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
  //For every blend function of every point stores the actual indexes of point of origin - more than one per point&blendfunc if point not in origin
  std::vector<std::vector<std::vector<int>>> refine_indexes;


  std::vector<std::vector<double>> origin_sarray, origin_tarray;
  std::vector<Vec> orig_cps;
  bool distMode = false, distColorMode = false, bringBackMode = false;
  std::vector<Vec> distColors;
  std::vector<double> fitDistances, orig_weights;
  std::vector<int> baseIndsInOrig, indsInOrig,IAOrig, JAOrig, rowsInOrig, colsInOrig;
  Eigen::MatrixXd M;
  std::vector<double> self_multiplier_for_temps;

  std::vector<Vec> sample_points, surface_points;
  bool draw_point_clouds = false;
  std::vector<double> us, vs, orig_us, orig_vs;
  std::vector<int> sample_corner_inds, fit_corner_inds;
  std::vector<double> distances;
  // new_point_added in order to look at actual N-R changes when checking whether needs new insertion
  bool sq_dist_mode = false, new_point_added = false;
  std::vector<double>::iterator max_dist_it;
  double last_max_dist, max_dist_change;
  const double max_dist_boundary = 0.03, max_distchange_boundary = 0.01;
  double last_sq_dist, sq_dist_change;
  const double sq_dist_boundary = 0.3, sq_distchange_boundary = 0.1;

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
