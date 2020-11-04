double MyViewer::getCutoffRatio() const {
  return cutoff_ratio;
}

void MyViewer::setCutoffRatio(double ratio) {
  cutoff_ratio = ratio;
  updateMeanMinMax();
}

double MyViewer::getMeanMin() const {
  return mean_min;
}

void MyViewer::setMeanMin(double min) {
  mean_min = min;
}

double MyViewer::getMeanMax() const {
  return mean_max;
}

void MyViewer::setMeanMax(double max) {
  mean_max = max;
}

const double *MyViewer::getSlicingDir() const {
  return slicing_dir.data();
}

void MyViewer::setSlicingDir(double x, double y, double z) {
  slicing_dir = Vector(x, y, z).normalized();
}

double MyViewer::getSlicingScaling() const {
  return slicing_scaling;
}

void MyViewer::setSlicingScaling(double scaling) {
  slicing_scaling = scaling;
}

double MyViewer::getSmoothingLambda() const {
	return smoothing_lambda;
}
void MyViewer::setSmoothingLambda(double lambda) {
	smoothing_lambda = lambda;
}

int MyViewer::getSampleNumOneD() const {
	return sample_num_1d;
}

void MyViewer::setSampleNumOneD(int sample_num) {
	sample_num_1d = sample_num;
}
