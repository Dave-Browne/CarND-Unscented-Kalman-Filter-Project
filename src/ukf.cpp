#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // Initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_ = false;

  // If this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // If this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // State dimension: [pos_x pos_y velocity_abs yaw_angle yaw_rate]
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = n_x_ + 2;

  // Number of sigma points
  n_sigma_ = (2 * n_aug_) + 1;

  // Initial state vector
  x_ = VectorXd(5);

  // Initial state covariance matrix
  // We have confidence in px & py initial values, but not v, yaw and yawd
  P_ = MatrixXd(5, 5);
  P_ << 0.5 , 0   , 0  , 0  , 0,
        0   , 0.5 , 0  , 0  , 0,
        0   , 0   , 1  , 0  , 0,
        0   , 0   , 0  , 0.5, 0,
        0   , 0   , 0  , 0  , 0.5;

  // Predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, n_sigma_);

  // Measurement noise matrix for laser
  R_laser_ = MatrixXd(2, 2);

  // Measurement noise matrix for radar
  R_radar_ = MatrixXd(3, 3);

  // Process noise standard deviation longitudinal acceleration in m/s^2 
  std_a_ = 0.83;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.55;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // Weights of sigma points
  weights_ = VectorXd(n_sigma_);
  for (int i=0; i<n_sigma_; i++) {
    if (i == 0)
      weights_(i) = lambda_ / (lambda_ + n_aug_);
    else
      weights_(i) = 0.5 / (lambda_ + n_aug_);
  }

  // The current NIS for radar
  NIS_radar_ = 0;

  // The current NIS for laser
  NIS_laser_ = 0;

  // Time when the state is true, in us, initialised randomly as 0
  time_us_ = 0;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package
 * The latest measurement data of either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  // If measurement data = 0, set state mean x_ = 0 and proceed to next measurement. Do not update time.
  if ((meas_package.raw_measurements_[0] == 0) & (meas_package.raw_measurements_[1] == 0)) {
    x_.fill(0.0);
    return;
  }

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    // First measurement
    cout << "std_a_ " << std_a_ << " std_yawdd " << std_yawdd_ << endl;
    x_.fill(0.0);

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert radar from polar to cartesian coordinates and initialize state.
      float rho = meas_package.raw_measurements_[0];
      float phi = meas_package.raw_measurements_[1];
      float px = rho * cos(phi);
      float py = rho * sin(phi);
      float v = 0;            // the radar measurement does not give us the CTRV velocity
      float yaw = 0;
      float yawd = 0;
      x_ << px, py, v, yaw, yawd;
      // Initialise the previous timestamp
      time_us_ = meas_package.timestamp_;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      // Set the state with the initial location and zero velocity
      float px = meas_package.raw_measurements_[0];
      float py = meas_package.raw_measurements_[1];
      x_ << px, py, 0, 0, 0;
      // Initialise the previous timestamp
      time_us_ = meas_package.timestamp_;
    }
    // Initialization complete, no need to do the ukf process
    is_initialized_ = true;
    return;
  }

  // Return if laser/radar not used
  if ((use_radar_ == 0) & (meas_package.sensor_type_ == MeasurementPackage::RADAR))
    return;
  else if ((use_laser_ == 0) & (meas_package.sensor_type_ == MeasurementPackage::LASER))
    return;

  // If not the first measurement, continue with the ukf prediction and update steps
  /*****************************************************************************
   *  Prediction
   ****************************************************************************/  
  // Compute the time elapsed between the current and previous measurements
  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;	//dt - expressed in seconds

  // If dt is large, divide it into smaller steps
  while (dt > 0.1) {
    const double dt_small = 0.05;
    Prediction(dt_small);
    dt -= dt_small;
  }
  // Store the current time in time_us_
  time_us_ = meas_package.timestamp_;

  Prediction(dt);
  // Print the output
//  cout << "Predicted x_ mean \n" << x_ << endl;
//  cout << "Predicted P_ covariance \n" << P_ << endl;


  /*****************************************************************************
   *  Update
   ****************************************************************************/
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  } 
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  }
  // Print the output
//  cout << "New x_ mean \n" << x_ << endl;
//  cout << "New P_ covariance \n" << P_ << endl;
}



/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

  /*****************************************************************************
   * 1. Generate Augmented Sigma points -> Xsig_aug
   ****************************************************************************/
  // Generate augmented x_ vector (state vector + noise vector)
  // Set first vars as x_
  VectorXd x_aug = VectorXd(n_aug_);                // x_aug 7
  x_aug.fill(0.0);
  x_aug.head(n_x_) = x_;

  // Generate augmented P_ matrix
  // Set top left corner a P_ and bottom right corner as noise covariance Q
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);        // P_aug 7x7
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug.bottomRightCorner(n_aug_-n_x_, n_aug_-n_x_) << pow(std_a_, 2), 0.0, 0.0, pow(std_yawdd_, 2);

  // Square root matrix A
  MatrixXd L = MatrixXd(n_aug_, n_aug_);            // L 7x7
  L.fill(0.0);
  L = P_aug.llt().matrixL();

  // Generate augmented sigma points matrix
  // Set the first column as x_aug
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sigma_);   // Xsig_aug 7x15
  Xsig_aug.fill(0.0);
  Xsig_aug.col(0) = x_aug;
  for (int i=0; i<n_aug_; i++) {
    Xsig_aug.col(i+1)        = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
  }

   /*****************************************************************************
   * 2. Predict Augmented Sigma points -> Xsig_pred_
   ****************************************************************************/
  for (int i=0; i<n_sigma_; i++) {
    double px = Xsig_aug(0,i);
    double py = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    // Predict sigma points
    double px_p, py_p;

    // Cater for division by zero
    if (fabs(yawd) > 0.001) {
      px_p = px + v/yawd * (sin(yaw + yawd*delta_t) - sin(yaw));
      py_p = py + v/yawd * (cos(yaw) - cos(yaw+yawd*delta_t));
    }
    else {
      px_p = px + v*delta_t*cos(yaw);
      py_p = py + v*delta_t*sin(yaw);
    }
    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;
    
    // Add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;
    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;
    
    // Write predicted sigma points into right column
    Xsig_pred_.col(i) << px_p, py_p, v_p, yaw_p, yawd_p;        // Xsig_pred_ 5x15
  }

   /*****************************************************************************
   * 3. Predict mean and covariance of the sigma points -> x_ & P_
   ****************************************************************************/
  // Loop through each column of the predicted sigma points matrix
  x_ = Xsig_pred_ * weights_;
  P_.fill(0.0);
  
  // Predict state covariance matrix
  for (int i=0; i<n_sigma_; i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // Normalize angle between -PI and +PI
    NormalizeAngle(&x_diff(3));

    P_ += weights_(i) * x_diff * x_diff.transpose();  // P_ 5x5
  }
}



/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

  /*****************************************************************************
   * 4. Predicted sigma points mean and covariance in lidar measurement space -> z_pred & S
   ****************************************************************************/
  // Set measurement dimension, lidar can measure px and py
  int n_z = 2;
  
  // Create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, n_sigma_);              // Zsig 2x15

  // Mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);                      // z_pred 2
  z_pred.fill(0.0);
  
  // Measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);                      // S 2x2
  S.fill(0.0);
  
  // Transform sigma points into measurement space
  Zsig = Xsig_pred_.block(0, 0, n_z, n_sigma_);

  for (int i=0; i<n_sigma_; i++) {
    // Predicted measurement mean
    z_pred += weights_(i) * Zsig.col(i);
  }

  // Predict measurement covariance matrix S (without measurement noise R)
  for (int i=0; i<n_sigma_; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S += weights_(i) * z_diff * z_diff.transpose();
  }
  // Define measurement noise matrix R_laser_
  R_laser_ << pow(std_laspx_, 2), 0,
              0                 , pow(std_laspy_, 2);

  // Add noise to measurement covariance matrix S
  S = S + R_laser_;
  
   /*****************************************************************************
   * 5. Update lidar state -> x_ & P_
   ****************************************************************************/
  // Create vector for incoming radar measurement
  VectorXd z = VectorXd(n_z);                           // z 2
  z.fill(0.0);
  double px = meas_package.raw_measurements_[0];
  double py = meas_package.raw_measurements_[1];
  z << px, py;               //px, py in m

  // Create cross correlation matrix Tc for sigma points in state and measurement spaces
  MatrixXd Tc = MatrixXd(n_x_, n_z);                    // Tc 5x2
  Tc.fill(0.0);

  for (int i=0; i<n_sigma_; i++) {
    // Measurement difference
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // State difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // Normalize angle between -PI and +PI
    NormalizeAngle(&x_diff(3));

    // Calculate cross correlation matrix
    Tc += weights_(i) * x_diff * z_diff.transpose();
  }
  // Create matrix for Kalman gain
  MatrixXd K = MatrixXd(n_x_, n_z);
  // Calculate Kalman gain K
  K = Tc * S.inverse();

  // Measurement difference
  VectorXd z_diff = z - z_pred;

  // Update state mean and covariance matrix
  x_ = x_ + (K * z_diff);
  P_ = P_ - (K * S * K.transpose());

  // Calculate NIS value
  NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
}



/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  
  /*****************************************************************************
   * 4. Predicted sigma points mean and covariance in radar measurement space -> z_pred & S
   ****************************************************************************/
  // Set measurement dimension, radar can measure rho, phi, and rho_dot
  int n_z = 3;
  
  // Create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, n_sigma_);              // Zsig 3x15
  Zsig.fill(0.0);

  // Mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);                      // z_pred 3
  z_pred.fill(0.0);
  
  // Measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);                      // S 3x3
  S.fill(0.0);
  
  // Transform sigma points into measurement space
  for (int i=0; i<n_sigma_; i++) {
    double px = Xsig_pred_(0,i);
    double py = Xsig_pred_(1,i);
    double v = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    // Accomodate for divide by 0 error when calculating rho_dot
    float c1 = px*px + py*py;
    // If division by zero, set denominator to a small number
    while(fabs(c1) < 0.001){
      cout << "ro_dot - Error - Division by Zero... ";
      cout << "px=" << px << " and py=" << py << " ... adding 0.023 and continuing" << endl;
      px += 0.023;
      py += 0.023;
      c1 = px*px + py*py;
    }

    double rho = sqrt(c1);
    double phi = atan2(py, px);
    double rho_dot = (px*v*cos(yaw) + py*v*sin(yaw)) / rho;           // Caution divide by 0

    // Predicted sigma points in the measurement space
    Zsig.col(i) << rho, phi, rho_dot;

    // Predicted measurement mean
    z_pred += weights_(i) * Zsig.col(i);
  }

  // Predict measurement covariance matrix S (without measurement noise R)
  for (int i=0; i<n_sigma_; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // Normalize angle between -PI and +PI
    NormalizeAngle(&z_diff(1));

    S += weights_(i) * z_diff * z_diff.transpose();
  }
  // Define measurement noise matrix R_radar_
  R_radar_ << pow(std_radr_, 2), 0                  , 0,
              0                , pow(std_radphi_, 2), 0,
              0                , 0                  , pow(std_radrd_, 2);

  // Add noise to measurement covariance matrix S
  S = S + R_radar_;
  
   /*****************************************************************************
   * 5. Update radar state -> x_ & P_
   ****************************************************************************/
  // Create vector for incoming radar measurement
  VectorXd z = VectorXd(n_z);                           // z 3
  z.fill(0.0);
  double rho = meas_package.raw_measurements_[0];
  double phi = meas_package.raw_measurements_[1];
  double rho_dot = meas_package.raw_measurements_[2];
  z << rho, phi, rho_dot;               //rho in m, phi in rad, rho_dot in m/s

  // Create cross correlation matrix Tc for sigma points in state and measurement spaces
  MatrixXd Tc = MatrixXd(n_x_, n_z);                    // Tc 5x3
  Tc.fill(0.0);

  for (int i=0; i<n_sigma_; i++) {
    // Measurement difference
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // Normalize angle between -PI and +PI
    NormalizeAngle(&z_diff(1));

    // State difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // Normalize angle between -PI and +PI
    NormalizeAngle(&x_diff(3));

    // Calculate cross correlation matrix
    Tc += weights_(i) * x_diff * z_diff.transpose();
  }
  // Create matrix for Kalman gain
  MatrixXd K = Tc * S.inverse();

  // Measurement difference
  VectorXd z_diff = z - z_pred;
  // Normalize angle between -PI and +PI
  NormalizeAngle(&z_diff(1));

  // Update state mean and covariance matrix
  x_ = x_ + (K * z_diff);
  P_ = P_ - (K * S * K.transpose());

  // Calculate NIS value
  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}


/**
 * Normalizes the angle
 */
void UKF::NormalizeAngle(double *pValue) {
  if (fabs(*pValue) > M_PI)
  {
    *pValue = fmod( *pValue, M_PI );
  }
}