
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>



#include <string>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <algorithm> 
#include <string>
#include <vector>
#include <sstream>
#include <functional>
#include <time.h> 
#include <cmath>
#include <climits>
#include <memory.h>

#include <curand.h>
#include <curand_kernel.h>

# define M_PI   3.14159265358979323846
# define G1n	1000
# define G2n	1000
# define Gn	G1n+G2n+1
# define DF_nnn 2000
# define theta_min 0
# define theta_max 180
# define theta_step 5
# define phi_min 0
# define phi_max 360
# define phi_step 5

using namespace std;

struct tmins
{
	double t, min_dist, min_vel, rv7;
};
struct coe
{
	double e, a, i, o, w, v;
};
struct vec
{
	double x, y, z;
	__device__ vec()
	{
		x = y = z = 0;
	}
	__device__ vec(double ix, double iy, double iz)
	{
		x = ix;
		y = iy;
		z = iz;
	}
	__device__ inline vec operator+(vec a) {
		vec r;
		r.x = a.x + x;
		r.y = a.y + y;
		r.z = a.z + z;
		return r;
	}
	__device__ inline vec operator-() {
		vec r;
		r.x = -x;
		r.y = -y;
		r.z = -z;
		return r;
	}
	__device__ inline vec operator-(vec a) {
		vec r;
		r = *this + -a;
		return r;
	}
	__device__ double dist() {
		return sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));
	}
	__device__ double sqrd() {
		return (pow(x, 2) + pow(y, 2) + pow(z, 2));
	}
	__device__ double dot(vec a) {
		return (a.x*x + a.y*y + a.z*z);
	}
	__device__ double mag(vec a) {
		return sqrt(this->dot(a));
	}
	__device__ vec cross(vec a) {
		vec r;
		r.x = y*a.z - z*a.y;
		r.y = z*a.x - x*a.z;
		r.z = x*a.y - y*a.x;
		return r;
	}
	__device__ vec scale(double sc) {
		vec r;
		r = *this*sc;
		return r;
	}
	__device__ inline vec operator*(vec a) {
		vec r;
		r.x = a.x * x;
		r.y = a.y * y;
		r.z = a.z * z;
		return r;
	}
	__device__ inline vec operator*(double a) {
		vec r;
		r.x = a * x;
		r.y = a * y;
		r.z = a * z;
		return r;
	}
	__device__ inline vec operator/(double a) {
		vec r;
		r.x = x / a;
		r.y = y / a;
		r.z = z / a;
		return r;
	}
	__device__ inline vec operator=(double a) {
		vec r;
		r.x = a;
		r.y = a;
		r.z = a;
		return r;
	}
};
struct pos_vel
{
	vec r;
	vec v;
	__device__ inline pos_vel operator+(pos_vel a) {
		pos_vel ret;
		ret.r = a.r + r;
		ret.v = a.v + v;
		return ret;
	}
	__device__ inline pos_vel operator*(pos_vel a) {
		pos_vel ret;
		ret.r = a.v * r;
		ret.v = a.v * v;
		return ret;
	}
	__device__ inline pos_vel operator*(double a) {
		pos_vel ret;
		ret.r = r * a;
		ret.v = v * a;
		return ret;
	}
	__device__ inline pos_vel operator/(double a) {
		pos_vel ret;
		ret.r = r / a;
		ret.v = v / a;
		return ret;
	}
};
struct gparam
{
	int galaxy;
	double mass, eps, epsilon, rin, rout, theta, phi, heat;
	vec rscale;
	int opt;
	int n;
};
class parameters
{
public:
	double mass_gm = 1.98892e44;
	double mass_solar = 1.98892e33;
	double distance = 4.6285203749999994e22;
	double time_s = 2.733342473337471e15;
	double vel_unit = distance / time_s;
	double pc = 3.08568025e18;
	double kpc = pc * 1000.0;
	double year = 365.25 * 24.0 * 3600.0;
	double km = 1e5;
	double vel_km_sec = vel_unit / km;
	double a_mss = distance / (time_s * time_s) / 100.0;
	double a0_mks = 1.2e-10;
	double a0 = a0_mks / a_mss;
	double pi = 3.141592653589793;
	double hbase = 0.001;

	int potential_type = 0;

	int ndim = 3;

	gparam galaxy1, galaxy2;

	pos_vel x0[Gn], xout[Gn];
	//double** x0, xout;

	int n;

	double time, tstart, tend;
	double inclination_degree;
	double omega_degree;
	double rmin;
	double velocity_factor;
	double mratio;
	double secondary_size;
	pos_vel sec_vec;
	bool use_sec_vec, tIsSet;

	double h;
	int nstep;
	int nout;

	int iout;
	int unit;
	int istep;

	curandState state;

	__device__ void standard_galaxy_both()
	{
		standard_galaxy(galaxy1, 1);
		standard_galaxy(galaxy2, 2);
	}

	__device__ void standard_galaxy(gparam& g, int galaxy)
	{
		g.galaxy = galaxy;
		g.mass = 1.0;
		g.epsilon = 0.3;
		g.rin = 0.05;
		g.rout = 1.0;
		g.rscale.x = 3.0;
		g.rscale.y = 3.0;
		g.rscale.z = 3.0;
		g.theta = 0.0;
		g.phi = 0.0;
		g.opt = 1;
		g.heat = 0.0;
		g.n = 1000;
	}

	__device__ void test_collision() {
		inclination_degree = 90.0;
		omega_degree = 0.0;
		rmin = 1.0;
		velocity_factor = 1.0;
		time = -3.0;

		h = hbase;
		nout = 5;
		nstep = 500;

		n = galaxy1.n + galaxy2.n;
	}

	__device__ vec unrotate_frame(vec in, double stheta, double ctheta, double sphi, double cphi) {
		vec r;

		r.x = in.x * ctheta + in.z * stheta;
		r.y = in.y;
		r.z = -in.x * stheta + in.z * ctheta;

		r.x = r.x * cphi - r.y * sphi;
		r.y = r.x * sphi + r.y * cphi;

		return r;
	}

	__device__ vec rotate_frame(vec in, double stheta, double ctheta, double sphi, double cphi) {
		vec r;

		r.x = in.x * cphi + in.y * sphi;
		r.y = -in.x * sphi + in.y * cphi;
		r.z = in.z;

		r.x = r.x * ctheta - r.z * stheta;
		r.z = r.x * stheta + r.z * ctheta;

		return r;
	}
	void input_particles(ifstream stream) {
		for (int i = 0; i < n; i++) {
			stream >> x0[i].r.x;
			stream >> x0[i].r.y;
			stream >> x0[i].r.z;
			stream >> x0[i].v.x;
			stream >> x0[i].v.y;
			stream >> x0[i].v.z;

		}

	}
	void output_particles(FILE* file, bool header_on) {
		if (header_on)
		{
			//fprintf(file, "%16.8f\n", time);
			//fprintf(file, "%16.8f%16.8f\n", galaxy1.mass, galaxy2.mass);
			//fprintf(file, "%16.8f%16.8f\n", galaxy1.eps, galaxy2.eps);
			fprintf(file, "n:%8i n1:%8i n2:%8i theta1:%16.8f phi1:%16.8f\n", n, galaxy1.n, galaxy2.n, galaxy1.theta, galaxy1.phi);
		}
		for (int i = 0; i < n; i++)
			fprintf(file, "%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f\n", x0[i].r.x, x0[i].r.y, x0[i].r.z, x0[i].v.x, x0[i].v.y, x0[i].v.z);
	}
	void create_gnuplot_script()
	{
		FILE *file;
		double xmin, xmax, ymin, ymax;
		double amax;
		//xmin = *min_element(x0[1], x0[1] + iout);
		//xmax = *max_element(x0[1], x0[1] + iout);
		//ymin = *min_element(x0[2], x0[2] + iout);
		//ymax = *max_element(x0[2], x0[2] + iout);
		xmin = minx(x0, iout);
		xmax = maxx(x0, iout);
		ymin = miny(x0, iout);
		ymax = maxy(x0, iout);

		amax = max(-xmin, xmax);
		amax = max(amax, -ymin);
		amax = max(amax, ymax);

		file = fopen("gscript", "w+");
		fprintf(file, "%s%15.6f%s%15.6f%s", "set xrange[,", -amax, ":", amax, "]");
		fprintf(file, "%s%15.6f%s%15.6f%s", "set yrange[,", -amax, ":", amax, "]");
		for (int i = 0; i < iout; i++)
			fprintf(file, "%s%3i%s", "plot 'a.", i, "' using 1:2");
		fclose(file);
	}
	double minx(pos_vel* pv, int n)
	{
		double min = INT_MAX;
		for (int i = 0; i < n; i++)
			if (pv[i].r.x < min)
				min = pv[i].r.x;
		return min;
	}
	double miny(pos_vel* pv, int n)
	{
		double min = INT_MAX;
		for (int i = 0; i < n; i++)
			if (pv[i].r.y < min)
				min = pv[i].r.y;
		return min;
	}
	double maxx(pos_vel* pv, int n)
	{
		double max = INT_MIN;
		for (int i = 0; i < n; i++)
			if (pv[i].r.x > max)
				max = pv[i].r.x;
		return max;
	}
	double maxy(pos_vel* pv, int n)
	{
		double max = INT_MIN;
		for (int i = 0; i < n; i++)
			if (pv[i].r.y > max)
				max = pv[i].r.y;
		return max;
	}
	void print_profile(int galaxy_num)
	{
		gparam g;
		if (galaxy_num == 1)
			g = galaxy1;
		else
			g = galaxy2;

		cout << "----------------------------------";
		cout << "GALAXY =" << g.galaxy;
		cout << "mass        = " << g.mass;
		cout << "epsilon     = " << g.eps;
		cout << "rin         = " << g.rin;
		cout << "rout        = " << g.rout;
		cout << "rscale      = " << g.rscale.x;
		cout << "rscale      = " << g.rscale.y;
		cout << "rscale      = " << g.rscale.z;
		cout << "theta       = " << g.theta;
		cout << "phi         = " << g.phi;
		cout << "opt         = " << g.opt;
		cout << "heat        = " << g.heat;
		cout << "particles   = " << g.n;
		cout << "----------------------------------";
	}
	void print_collision()
	{
		cout << "----------------------------------";
		cout << "COLLISION PARAMETERS";
		cout << "n           = " << n;
		cout << "time        = " << time;
		cout << "inclination = " << inclination_degree;
		cout << "omega       = " << omega_degree;
		cout << "rmin        = " << rmin;
		cout << "velocity    = " << velocity_factor;
		cout << "h           = " << h;
		cout << "nstep       = " << nstep;
		cout << "nout        = " << nout;
		cout << "----------------------------------";
	}
	void octave_parameters_out(pos_vel pv, pos_vel x00)
	{
		cout << "$mass1 = " << galaxy1.mass << ";" << endl;
		cout << "$t1    = " << galaxy1.theta << ";" << endl;
		cout << "$p1    = " << galaxy1.phi << ";" << endl;
		cout << "$rout1 = " << galaxy1.rout << ";" << endl;
		cout << "$mass2 = " << galaxy2.mass << ";" << endl;
		cout << "$t2    = " << galaxy2.theta << ";" << endl;
		cout << "$p2    = " << galaxy2.phi << ";" << endl;
		cout << "$rout2 = " << galaxy2.rout << ";" << endl;
		cout << "$xf    = " << pv.r.x << ";" << endl;
		cout << "$yf    = " << pv.r.y << ";" << endl;
		cout << "$zf    = " << pv.r.z << ";" << endl;
		cout << "$vxf   = " << pv.v.x << ";" << endl;
		cout << "$vyf   = " << pv.v.y << ";" << endl;
		cout << "$vzf   = " << pv.v.z << ";" << endl;
		cout << "$x     = " << x00.r.x << ";" << endl;
		cout << "$y     = " << x00.r.y << ";" << endl;
		cout << "$z     = " << x00.r.z << ";" << endl;
		cout << "$vx    = " << x00.v.x << ";" << endl;
		cout << "$vy    = " << x00.v.y << ";" << endl;
		cout << "$vz    = " << x00.v.z << ";" << endl;
		cout << "$t     = " << tend << ";" << endl;
	}
	void read_parameter_file(ifstream& in)
	{
		string line, label;
		double val;
		while (!in.eof()) {
			line.clear();
			label.clear();
			val = 0;
			in >> line;
			split_str(line, label, val);
			if (label == "potential_type")
				potential_type = (int)val;
			else if (label == "mass1")
				galaxy1.mass = val;
			else if (label == "mass2")
				galaxy2.mass = val;
			else if (label == "epsilon1")
				galaxy1.epsilon = val;
			else if (label == "epsilon2")
				galaxy2.epsilon = val;
			else if (label == "rin1")
				galaxy1.rin = val;
			else if (label == "rin2")
				galaxy2.rin = val;
			else if (label == "rout1")
				galaxy1.rout = val;
			else if (label == "rout2")
				galaxy2.rout = val;
			else if (label == "theta1")
				galaxy1.theta = val;
			else if (label == "theta2")
				galaxy2.theta = val;
			else if (label == "phi1")
				galaxy1.phi = val;
			else if (label == "phi2")
				galaxy2.phi = val;
			else if (label == "opt1")
				galaxy1.opt = (int)val;
			else if (label == "opt2")
				galaxy2.opt = (int)val;
			else if (label == "heat1")
				galaxy1.heat = val;
			else if (label == "heat2")
				galaxy2.heat = val;
			else if (label == "n1")
				galaxy1.n = (int)val;
			else if (label == "n2")
				galaxy2.n = (int)val;
			else if (label == "inclination_degree")
				inclination_degree = val;
			else if (label == "omega_degree")
				omega_degree = val;
			else if (label == "rmin")
				rmin = val;
			else if (label == "velocity_factor")
				velocity_factor = val;
			else if (label == "tstart") {
				time = val;
				tstart = val;
				tIsSet = true;
			}
			else if (label == "tend")
				tend = val;
			else if (label == "h")
				h = val;
			else if (label == "rx") {
				use_sec_vec = true;
				sec_vec.r.x = val;
			}
			else if (label == "ry") {
				use_sec_vec = true;
				sec_vec.r.y = val;
			}
			else if (label == "rz") {
				use_sec_vec = true;
				sec_vec.r.z = val;
			}
			else if (label == "vx") {
				use_sec_vec = true;
				sec_vec.v.x = val;
			}
			else if (label == "vy") {
				use_sec_vec = true;
				sec_vec.v.y = val;
			}
			else if (label == "vz") {
				use_sec_vec = true;
				sec_vec.v.z = val;
			}
			else if (label == "rscale11")
				galaxy1.rscale.x = val;
			else if (label == "rscale12")
				galaxy1.rscale.y = val;
			else if (label == "rscale13")
				galaxy1.rscale.z = val;
			else if (label == "rscale21")
				galaxy2.rscale.x = val;
			else if (label == "rscale22")
				galaxy2.rscale.y = val;
			else if (label == "rscale23")
				galaxy2.rscale.z = val;
			else
				cout << "skipping line ";
		}
	}
	void split_str(string in, string& label, double& val)
	{
		string sval;
		int ind, len;
		char strt = in.at(0);
		if (strt == '!' || strt == '#' || strt == '/') {
			label = "!";
			val = 0;
			return;
		}
		ind = in.find("=");
		if (ind == 0) {
			label = "!";
			val = 0;
		}
		label = in.substr(0, ind);
		len = in.length() - label.length();
		sval = in.substr(ind + 1, len);
		val = strtod(sval.c_str(), NULL);
	}

	void write_parameter_file(ofstream o)
	{
		o << "potential_type=" << potential_type;
		o << "mass1=" << galaxy1.mass;
		o << "mass2=" << galaxy2.mass;
		o << "epsilon1=" << galaxy1.epsilon;
		o << "epsilon2=" << galaxy2.epsilon;
		o << "rin1=" << galaxy1.rin;
		o << "rin2=" << galaxy2.rin;
		o << "rout1=" << galaxy1.rout;
		o << "rout2=" << galaxy2.rout;
		o << "theta1=" << galaxy1.theta;
		o << "theta2=" << galaxy2.theta;
		o << "phi1=" << galaxy1.phi;
		o << "phi2=" << galaxy2.phi;
		o << "opt1=" << galaxy1.opt;
		o << "opt2=" << galaxy2.opt;
		o << "heat1=" << galaxy1.heat;
		o << "heat2=" << galaxy2.heat;
		o << "n1=" << galaxy1.n;
		o << "n2=" << galaxy2.n;
		o << "inclination_degree=" << inclination_degree;
		o << "omega_degree=" << omega_degree;
		o << "rmin=" << rmin;
		o << "velocity_factor=" << velocity_factor;
		o << "tstart=" << tstart;
		o << "tend=" << tend;
		o << "h=" << h;
		o << "rx=" << sec_vec.r.x;
		o << "ry=" << sec_vec.r.y;
		o << "rz=" << sec_vec.r.z;
		o << "vx=" << sec_vec.v.x;
		o << "vy=" << sec_vec.v.y;
		o << "vz=" << sec_vec.v.z;
		o << "rscale11=" << galaxy1.rscale.x;
		o << "rscale12=" << galaxy1.rscale.y;
		o << "rscale13=" << galaxy1.rscale.z;
		o << "rscale21=" << galaxy2.rscale.x;
		o << "rscale22=" << galaxy2.rscale.y;
		o << "rscale23=" << galaxy2.rscale.z;
	}

	__device__ void set_state_info(double* infos)
	{
		potential_type = 0;
		sec_vec.r.x = infos[1];
		sec_vec.r.y = infos[2];
		sec_vec.r.z = infos[3];
		sec_vec.v.x = infos[4];
		sec_vec.v.y = infos[5];
		sec_vec.v.z = infos[6];
		galaxy1.mass = infos[7];
		galaxy2.mass = infos[8];
		galaxy1.rout = infos[9];
		galaxy2.rout = infos[10];
		galaxy1.phi = infos[11];
		galaxy2.phi = infos[12];
		galaxy1.theta = infos[13];
		galaxy2.theta = infos[14];
		galaxy1.epsilon = infos[15];
		galaxy2.epsilon = infos[16];
		galaxy1.rscale.x = infos[17];
		galaxy1.rscale.y = infos[18];
		galaxy1.rscale.z = infos[19];
		galaxy2.rscale.x = infos[20];
		galaxy2.rscale.y = infos[21];
		galaxy2.rscale.z = infos[22];
		use_sec_vec = true;
	}

	__device__ void parse_state_info_string(string in)
	{
		istringstream ss(in);
		string token;
		double infos[23];
		int i = 0;
		while (getline(ss, token, ',')) {
			infos[i] = strtod(token.c_str(), NULL);
		}
		potential_type = 0;
		sec_vec.r.x = infos[1];
		sec_vec.r.y = infos[2];
		sec_vec.r.z = infos[3];
		sec_vec.v.x = infos[4];
		sec_vec.v.y = infos[5];
		sec_vec.v.z = infos[6];
		galaxy1.mass = infos[7];
		galaxy2.mass = infos[8];
		galaxy1.rout = infos[9];
		galaxy2.rout = infos[10];
		galaxy1.phi = infos[11];
		galaxy2.phi = infos[12];
		galaxy1.theta = infos[13];
		galaxy2.theta = infos[14];
		galaxy1.epsilon = infos[15];
		galaxy2.epsilon = infos[16];
		galaxy1.rscale.x = infos[17];
		galaxy1.rscale.y = infos[18];
		galaxy1.rscale.z = infos[19];
		galaxy2.rscale.x = infos[20];
		galaxy2.rscale.y = infos[21];
		galaxy2.rscale.z = infos[22];
		use_sec_vec = true;
	}
};
class df_module
{
public:
	const static int nnn = DF_nnn;
	double rad[nnn];
	double rho_halo[nnn], mass_halo[nnn];
	double rho_disk[nnn], mass_disk[nnn];
	double rho_bulge[nnn], mass_bulge[nnn];
	double rho_total[nnn], mass_total[nnn];
	double masses[nnn], radius[nnn], density[nnn];
	double vr2[nnn], vr[nnn], new_vr2[nnn], new_vr[nnn];
	double acceleration[nnn], acceleration_particle[nnn];
	double new_mass[nnn], new_rho[nnn], phi[nnn];

	double rs_internal = 10.0;

	double rs2 = rs_internal * rs_internal;
	double rs3 = rs2 * rs_internal;

	double pscale;
	double lnl;

	__device__ void init_distribution()
	{
		double rmax;
		double mold, dmold, mtot;
		double rscale;
		double dx, x;
		double alphahalo, qhalo, gammahalo, mhalo, rchalo, rhalo, epsilon_halo;
		double zdisk, hdisk, zdiskmax;
		double hbulge, mbulge;
		double rho_tmp;
		double G, factor;
		double r, m, sqrtpi;
		double p1, rd, rho_local;
		double p, rr, dr, rh, dp, mnew, dm;
		double acc_merge, rad_merge, acc;
		double pi = M_PI;

		int j, nmax, k, nmerge, ntotal, jj;

		//set the constant for dynamical friction
		//lnl = 0.00;
		//default for Merger Zoo
		lnl = 0.001;

		//set up the parameters for the halo
		mhalo = 5.8;
		rhalo = 10.0;
		rchalo = 10.0;
		gammahalo = 1.0;
		epsilon_halo = 0.4;
		sqrtpi = sqrt(pi);
		//////////
		//derive additional constants
		qhalo = gammahalo / rchalo;
		alphahalo = 1.0 / (1.0 - sqrtpi * qhalo * exp(pow(qhalo, 2)) * (1.0 - erf(qhalo)));
		//////////
		//set the integration limits and zero integration constants
		rmax = 20;
		nmax = 2000;
		dr = rmax / (nmax);
		mold = 0;
		rscale = 5;
		//ntotal = nmax * rscale;
		ntotal = nnn;
		//////////
		//set the limits for integration, and zero integration constants
		k = nmax / 2;
		dx = 1.0 / k;
		x = 0.0;
		dmold = 0.0;
		mtot = 0.0;
		//rad = 0.0;
		memset(rad, 0, 4 * nnn);
		m = 0.0;
		G = 1;
		//////////
		//set the fundamental disk parameters
		zdisk = 0.2;
		zdiskmax = 3.5;
		hdisk = 1.0;
		//////////
		//set the fundamental bulge parameters
		hbulge = 0.2;
		mbulge = 0.3333;
		//////////////////////////////////////////////////////////////////////////////////////////////
		//////////set up the radius array
		for (j = 0; j < nmax; j++) {
			x = x + dx;
			rad[j] = x * rchalo;
		}
		//////////////////////////////////////////////////////////////////////////////////////////////
		//////////
		dr = rad[2] - rad[1];
		dx = dr / rchalo;
		for (j = 0; j < nmax; j++) {
			//set the position
			r = rad[j];
			x = r / rchalo;
			//calculate the local rho based
			rho_tmp = alphahalo / (2 * pow(sqrtpi, 3)) * (exp(pow(-x, 2)) / (pow(x, 2) + pow(qhalo, 2)));
			//renormalize for the new halo size
			rho_tmp = rho_tmp / (rchalo * rchalo * rchalo);
			//calculate mass in local shell, and update total mass
			//dm = rho_tmp * 4 * pi * x * x *dx
			dm = rho_tmp * 4 * pi * r * r *dr;
			mtot = mtot + dm;
			//store values in an array
			rho_halo[j] = rho_tmp * mhalo;
			mass_halo[j] = mtot * mhalo;
		}
		//////////
		//now calculate the potential
		for (j = 0; j < nmax; j++) {
			r = rad[j];
			m = mass_halo[j];
			p1 = -G * m / r;
			phi[j] = p1;
		}
		//////////////////////////////////////////////////////////////////////////////////////////////
		//disk model
		//////////
		//loop over the distribution
		for (j = 0; j < nmax; j++) {
			//set the radius
			rd = rad[j];
			//find the local density in the disk
			rho_local = exp(-rd / hdisk) / (8 * pi*pow(hdisk, 2.0));
			rho_disk[j] = rho_local;
			//find the mass in the spherical shell
			mnew = 4 * pi * rho_local * rd *rd * dr;
			mass_disk[j] = mnew + mold;
			mold = mass_disk[j];
		}
		//////////////////////////////////////////////////////////////////////////////////////////////
		//bulge model
		//////////
		//loop over the distribution
		mold = 0.0;
		for (j = 0; j < nmax; j++) {
			//set the radius
			rd = rad[j];
			//find the local density in the disk
			rho_local = exp(pow(-rd, 2) / pow(hbulge, 2));
			rho_bulge[j] = rho_local;
			//find the mass in the spherical shell
			mnew = 4 * pi * rho_local * rd *rd * dr;
			mass_bulge[j] = mnew + mold;
			mold = mass_bulge[j];
		}
		//renormalize distribution
		factor = mbulge / mass_bulge[nmax];
		for (j = 0; j < nmax; j++) {
			mass_bulge[j] = mass_bulge[j] * factor;
			rho_bulge[j] = rho_bulge[j] * factor;
		}
		dr = rad[2] - rad[1];
		//////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////
		j = 1;
		mass_total[j] = (mass_halo[j] + mass_disk[j] + mass_bulge[j]);
		r = rad[j];
		rho_total[j] = mass_total[j] / (4.0 / 3.0 * pi * r * r * r);
		dr = rad[2] - rad[1];
		for (j = 1; j < nmax; j++) {
			r = rad[j];
			mass_total[j] = (mass_halo[j] + mass_disk[j] + mass_bulge[j]);
			dm = mass_total[j] - mass_total[j - 1];
			rho_total[j] = dm / (4 * pi * r * r * dr);
		}
		////////////////////////////////////////////////////////////////////////////////////////////
		//find the velocity dispersion pow(v_r,2)
		//masses = mass_total;
		//radius = rad;
		//density = rho_total;

		memcpy(masses, mass_total, 4 * nnn);
		memcpy(radius, rad, 4 * nnn);
		memcpy(density, rho_total, 4 * nnn);

		for (j = 0; j < nmax; j++) {
			p = 0.0;
			rr = radius[j];
			dr = radius[nmax] / nmax;
			for (jj = j; jj < nmax; jj++) {
				m = masses[jj];
				rh = density[jj];
				rr = rr + dr;
				dp = rh * G * m / pow(rr, 2) * dr;
				p = p + dp;
			}
			vr2[j] = 1 / density[j] * p;
			vr[j] = sqrt(vr2[j]);
		}
		////////////////////////////////////////////////////////////////////////////////////////////
		//find the velocity dispersion pow(v_r,2)
		//masses = mass_total;
		//radius = rad;
		//density = rho_total;

		memcpy(masses, mass_total, 4 * nnn);
		memcpy(radius, rad, 4 * nnn);
		memcpy(density, rho_total, 4 * nnn);

		for (j = 0; j < nmax; j++) {
			p = 0.0;
			rr = radius[j];
			dr = radius[nmax] / nmax;
			for (jj = j; jj < nmax; jj++) {
				m = masses[jj];
				rh = density[jj];
				rr = rr + dr;
				dp = rh * G * m / pow(rr, 2) * dr;
				p = p + dp;
			}
			vr2[j] = 1 / density[j] * p;
			vr[j] = sqrt(vr2[j]);
		}
		////////////////////////////////////////////////////////////////////////////////////////////
		//find the accelerations felt by the particles and center of mass
		//masses = mass_total;
		//radius = rad;
		//density = rho_total;

		memcpy(masses, mass_total, 4 * nnn);
		memcpy(radius, rad, 4 * nnn);
		memcpy(density, rho_total, 4 * nnn);

		for (j = 0; j < nmax; j++) {
			rr = radius[j];
			m = masses[j];
			acceleration[j] = G * m / pow(rr, 2);
		}
		//acceleration_particle = acceleration;
		memcpy(acceleration_particle, acceleration, 4 * nnn);

		nmerge = 50;
		acc_merge = acceleration[nmerge];
		rad_merge = rad[nmerge];
		for (j = 0; j < nmerge; j++) {
			rr = radius[j];
			m = masses[j];
			//smoothed acceleration
			acc = G * m / (pow(rr, 2) + .1* (rad_merge - rr));
			acceleration_particle[j] = acc;
		}
		////////////////////////////////////////////////////////////////////////////////////////////
		//rederive the masses from the new particle acceleration
		//radius = rad;
		memcpy(radius, rad, 4 * nnn);

		dr = rad[2] - rad[1];
		//find the accelerations felt by the particles and center of mass
		memcpy(radius, rad, 4 * nnn);

		for (j = 0; j < nmax; j++) {
			rr = radius[j];
			new_mass[j] = pow(rr, 2) * acceleration_particle[j] / G;
			new_rho[j] = new_mass[j] / (4 * pi * rr * rr * dr);
		}
		////////////////////////////////////////////////////////////////////////////////////////////
		//find the velocity dispersion pow(v_r,2) using the new density and masses


		//masses = new_mass;
		//radius = rad;
		//density = new_rho;

		memcpy(masses, new_mass, 4 * nnn);
		memcpy(radius, rad, 4 * nnn);
		memcpy(density, new_rho, 4 * nnn);

		for (j = 0; j < nmax; j++) {
			p = 0.0;
			rr = radius[j];
			dr = radius[nmax] / nmax;
			for (jj = j; jj < nmax; jj++) {
				m = masses[jj];
				rh = density[jj];
				rr = rr + dr;
				dp = rh * G * m / pow(rr, 2) * dr;
				p = p + dp;
			}
			new_vr2[j] = 1 / density[j] * p;
			new_vr[j] = sqrt(new_vr2[j]);
		}
		////////////////////////////////////////////////////////////////////////////////////////////
		//extend the values to large rmax
		for (j = nmax + 1; j < ntotal; j++) {
			mass_total[j] = mass_total[nmax];
			mass_halo[j] = mass_halo[nmax];
			mass_disk[j] = mass_disk[nmax];
			mass_bulge[j] = mass_bulge[nmax];
			new_mass[j] = new_mass[nmax];
			//rho_total[j] = 1e-3;
			//new_rho[j] = new_rho[nmax];
			rho_total[j] = 0.0;
			new_rho[j] = 0.0;
			vr[j] = 1e-6;
			vr2[j] = 1e-6;
			new_vr[j] = 1e-6;
			new_vr2[j] = 1e-6;
			m = mass_total[nmax];
			rr = rad[nmax] + dr*(j - nmax);
			rad[j] = rr;
			acc = G * m / pow(rr, 2);
			acceleration_particle[j] = acc;
			acceleration[j] = acc;
		}
		////////////////////////////////////////////////////////////////////////////////////////////
		//normalize to the unit mass
		for (j = 0; j < ntotal; j++) {
			mass_total[j] = mass_total[j] / 7.13333;
			mass_halo[j] = mass_halo[j] / 7.13333;
			mass_disk[j] = mass_disk[j] / 7.13333;
			mass_bulge[j] = mass_bulge[j] / 7.13333;
			new_mass[j] = new_mass[j] / 7.13333;
			rho_total[j] = rho_total[j] / 7.13333;
			new_rho[j] = new_rho[j] / 7.13333;
			vr[j] = vr[j] / 7.13333;
			vr2[j] = vr2[j] / 7.13333;
			new_vr[j] = new_vr[j] / 7.13333;
			new_vr2[j] = new_vr2[j] / 7.13333;
			rad[j] = rad[j];
			acceleration_particle[j] = acceleration_particle[j] / 7.13333;
			acceleration[j] = acceleration[j] / 7.13333;
			////write(11, *) rad[j], new_rho[j], new_mass[j], new_vr[j];
		}
		pscale = 1.0;
	}
	__device__ int df_index(double rin, double rs)
	{//                       why is rs here?
		double rmax_scale = 100.0;
		int local_nnn = nnn;
		int first_param = int((rin * pscale * rs_internal / rmax_scale) * nnn + 1);
		return min(first_param, local_nnn);
	}
};
class setup_module
{
public:
	df_module* df;
	double* t0;
	double phi_i1, phi_i2, theta_i1, theta_i2;
	vec rscale_i1, rscale_i2;
	double rrout1, rrout2;
	parameters *p;

	__device__ void wrap_rk41(pos_vel xx0, pos_vel& xxe)
	{
		if (p->potential_type == 0)
			xxe = rk41(xx0, &setup_module::diffq_spm);
		else if (p->potential_type == 1)
			xxe = rk41(xx0, &setup_module::diffq_nbi);
		else if (p->potential_type == 2)
			xxe = rk41(xx0, &setup_module::diffq_mond);
	}
	__device__ void perturber_position(pos_vel& original_rv)
	{
		double en, v1;
		pos_vel xx0;
		double omega, incl;
		double epsilon1, epsilon2;

		epsilon1 = sqrt(p->galaxy1.eps);
		epsilon2 = sqrt(p->galaxy2.eps);

		//change inclination and omega into radians
		incl = p->inclination_degree * M_PI / 180.0;
		omega = p->omega_degree * M_PI / 180.0;

		//energy from mass1
		if (p->galaxy1.epsilon > 0.0)
			en = p->galaxy1.mass / epsilon1 * (M_PI / 2.0 - atan(p->rmin / epsilon1));
		else
			en = p->galaxy1.mass / p->rmin;

		//energy from mass2
		if (p->galaxy2.epsilon > 0.0)
			en = p->galaxy2.mass / epsilon2 * (M_PI / 2.0 - atan(p->rmin / epsilon2));
		else
			en = p->galaxy2.mass / p->rmin;

		//calculate escape velocity and velocity at rmin
		v1 = sqrt(2.0 * en);
		v1 = sqrt(2.0)*circular_velocity(p->galaxy1.mass + p->galaxy2.mass, p->rmin,
			rrout1, p->galaxy1.epsilon, p->potential_type, p->a0);

		//adjust velocity for MOND
		v1 = -v1 * p->velocity_factor;


		//setup the transformaton based on the matrix in
		//fundimentals of astrodynamics p-> 82 by
		//bates, mueller, and white(1971)

		xx0.r.x = cos(omega) * p->rmin;
		xx0.r.y = sin(omega) * cos(incl) * p->rmin;
		xx0.r.z = sin(omega) * sin(incl) * p->rmin;

		xx0.v.x = -sin(omega) * v1;
		xx0.v.y = cos(omega) * cos(incl) * v1;
		xx0.v.z = cos(omega) * sin(incl) * v1;

		//update sec_vec
		p->sec_vec = xx0;
		p->sec_vec.v = -p->sec_vec.v;

		perturber_position_vec(xx0, original_rv);
	}
	__device__ void perturber_position_vec(pos_vel xx0, pos_vel& original_rv)
	{
		pos_vel xxe;
		int i;
		double dist1;
		double tcurrent;
		double epsilon1, epsilon2;

		epsilon1 = p->galaxy1.epsilon;
		epsilon2 = p->galaxy2.epsilon;

		//copy the original input vector
		original_rv = xx0;

		//reverse the velocity for backward integration 
		xx0.v = -xx0.v;

		//cout << xx0.print() << endl;

		//now move position back to t0 from t = 0.0
		tcurrent = 0;
		while (*t0 < tcurrent)
		{
			wrap_rk41(xx0, xxe);
			dist1 = xx0.r.dist();
			xx0 = xxe;
			tcurrent = tcurrent - p->h;
		}

		//reverse the velocity for forward integration 
		xx0.v = -xx0.v;

		//now adjust the test particles from the
		//second disk to the proper velocity and positions
		if (p->n > p->galaxy1.n)
			for (i = p->galaxy1.n; i < p->n; i++)
				p->x0[i] = p->x0[i] + xx0;// x0(i, :) = x0(i, :) + xx0(:);

										  //include the perturbing galaxy
										  //p->n += 1;
		p->x0[p->n] = xx0;
	}
	__device__ void reset_perturber_position(pos_vel pv, pos_vel& minloc, pos_vel& zcrossloc, double& tzcross)
	{
		pos_vel xx0, xxe;
		int i, istep;
		double tcurrent, dist, dist_old, zdist, zdist_old;
		bool zmin_flag, min_flag;
		int izmin;
		double rtime;

		//set the positions and velocity of the companion
		xx0.r = pv.r;
		xx0.v = -pv.v;

		//now move position back to t0 from t=0.0
		tcurrent = 0.0;
		p->tend = 0.0;
		istep = 0;
		rtime = 0.0;

		dist_old = 1.0e10;
		zdist_old = 1.0e10;
		min_flag = true;
		zmin_flag = true;

		while (*t0 < tcurrent)
		{

			wrap_rk41(xx0, xxe);

			//write(18, *) xx0

			dist = xx0.r.dist();
			xx0 = xxe;


			//if the distance is larger than the last step, update the
			//clock
			if (dist > dist_old)
			{
				tcurrent = tcurrent - p->h;

				//record the minimum location
				if (min_flag)
					minloc = xx0;
				min_flag = false;

			}
			else
			{
				//if the distance is larger thant he last step, update the
				//ending time of the simulation and the closest point
				p->tend += p->h;
				dist_old = dist;
			}


			//if the distance from the z plan is larger than the last
			//step, set the crossing location and time
			zdist = xx0.r.z;
			if (abs(zdist) > abs(zdist_old) && zmin_flag)
			{
				zmin_flag = false;
				zcrossloc = xx0;
				izmin = istep;
			}
			zdist_old = zdist;

			rtime = rtime + p->h;
			istep = istep + 1;

		}
		tzcross = p->tend - p->h * izmin;

		cout << "t0, tcurrent " << t0 << tcurrent;
		cout << "tend , istep " << p->tend << istep;
		cout << "rtime " << rtime;

		//set the time to t0
		xx0.v = -xx0.v;


		//now move adjust the test particles from the
		//second disk to the proper velocity and positions

		if (p->n > p->galaxy1.n)
			for (i = p->galaxy1.n + 1; i <= p->n; i++)
				p->x0[p->n] = p->x0[p->n] + xx0;

		//include the perturbing galaxy
		p->n += 1;
		p->x0[p->n] = xx0;

		///pscale = 1.1
	}
	__device__ pos_vel rk41(pos_vel xx0, pos_vel(setup_module::*diffq1)(pos_vel))
	{
		pos_vel x, f;
		pos_vel xxe;
		x = xx0;
		f = (this->*diffq1)(x);

		xxe = xx0 + f * p->h / 6.0;
		x = xx0 + f * p->h / 2.0;
		f = (this->*diffq1)(x);

		xxe = xxe + f * p->h / 3.0;
		x = xx0 + f * p->h / 2.0;
		f = (this->*diffq1)(x);

		xxe = xxe + f * p->h / 3.0;
		x = xx0 + f * p->h;
		f = (this->*diffq1)(x);

		xxe = xxe + f * p->h / 6.0;

		return xxe;
	}
	__device__ pos_vel diffq_spm(pos_vel x)
	{
		pos_vel r;
		double r21, r1, a1;
		r21 = x.r.sqrd();
		r1 = sqrt(r21);

		a1 = -p->galaxy1.mass / (r21 + p->galaxy1.eps) - p->galaxy2.mass / (r21 + p->galaxy2.eps);
		r.r = x.v;
		r.v = x.r / r1 * a1;

		return r;
	}
	__device__ pos_vel diffq_nbi(pos_vel x)
	{
		pos_vel r;
		double r21, r1, a1, a2, at;

		double c1, c2, c3, v21, v1, xvalue;
		double sqrtpi;

		int ival, ival2;
		double df_force1, df_force2;
		double df_sigma, df_rho;
		double ee1, ee2;

		//fix to eliminate a compilation warning message for unused variables
		ee1 = p->galaxy1.eps;
		ee2 = p->galaxy2.eps;

		sqrtpi = sqrt(M_PI);

		r21 = x.r.sqrd();
		r1 = sqrt(r21);

		//get the index for the calculations
		ival = df->df_index(r1, rrout1);
		ival2 = df->df_index(r1, rrout2);

		//get the forces, sigma and rho, and rescale them
		df_force1 = df->acceleration_particle[ival] * df->rs2;
		df_force2 = df->acceleration_particle[ival2] * df->rs2;

		df_sigma = df->new_vr[ival] * df->rs2;
		df_rho = df->new_rho[ival] * df->rs3;

		//interpolated forces 
		a1 = -p->galaxy1.mass * df_force1;
		a2 = -p->galaxy2.mass * df_force2;
		at = a1 + a2;

		//df
		v21 = x.v.sqrd();
		v1 = sqrt(v21);
		xvalue = v1 / df_sigma;
		c1 = erf(xvalue) - 2.0 * xvalue / sqrtpi * exp(-xvalue*xvalue);

		//df formula with G=1
		c2 = -4.0 * M_PI * p->galaxy2.mass * df->lnl / v21;
		c3 = c1 * c2 * df_rho;

		r.r = x.v;
		r.v = x.r / r1*at - x.v / v1*c3;

		return r;
	}
	__device__ pos_vel diffq_mond(pos_vel x)
	{
		pos_vel r;
		double r21, r1, a1, tmp, a2;

		r21 = x.r.sqrd();
		r1 = sqrt(r21);

		a1 = -p->galaxy1.mass / (r21 + p->galaxy1.eps);
		a2 = -p->galaxy2.mass / (r21 + p->galaxy2.eps);

		tmp = 2 * p->a0 / a1;
		a1 = a1 / sqrt(2.0) * sqrt(1.0 + sqrt(1.0 + tmp*tmp));

		tmp = 2 * p->a0 / a2;
		a2 = a2 / sqrt(2.0) * sqrt(1.0 + sqrt(1.0 + tmp*tmp));

		a1 = a1 + a2;

		r.r = x.v;
		r.v = x.r / r1*a1;

		return r;
	}
	__device__ void profile()
	{
		profile_g(p->galaxy1, 0);
		profile_g(p->galaxy2, p->galaxy1.n);


	}
	__device__ void profile_g(gparam g, int nstart)
	{
		//variables -
		//opt - option for the distribution
		//rin - inner radius
		//rout - outer radius
		//rscale - scale of brightness drop
		//nstart - start number for placement of particles
		//ntot - number of particles to be placed
		//heat - heat parameter
		//m - mass of galaxy
		//sl - softening length
		//nring - number of rings
		//npart - number of particle per ring(opt)
		//x0 - position of center of mass

		double stheta, ctheta, sphi, cphi;
		double x3, y3, z3, xv3, yv3, zv3, x2, y2, z2, xv2, yv2, zv2;
		double x, y, z, xv, yv, zv;
		int i, j, n;
		double rnorm;
		//double* rp, *r, *angle, *v, *p_ring, *cp_ring, *n_ring;
		double st, ct, dr, ran, r1, r2, ptot;
		int nring, dnring, is, ie, iring, tct;

		double ntot = nstart + g.n;

		n = p->n;
		double r[Gn], angle[Gn], v[Gn];

		stheta = sin(g.theta*M_PI / 180.0);
		ctheta = cos(g.theta*M_PI / 180.0);
		sphi = sin(g.phi*M_PI / 180.0);
		cphi = cos(g.phi*M_PI / 180.0);

		//set up the probablity distribution for the disk

		const int nprof = 1000;
		nring = nprof / 10;

		dnring = nprof / nring;
		double rp[nprof], n_ring[nprof], p_ring[nprof], cp_ring[nprof];

		//set the differential sum of the probability function into a vector
		rnorm = 0.0;
		dr = (g.rout - g.rin) / float(nprof);
		for (i = 0; i < nprof; i++)
		{
			r1 = float(i)*dr + g.rin;
			rp[i] = distrb(r1, g.opt, g.rscale) * r1 * dr * 2.0 * M_PI;
			rnorm = rnorm + rp[i];
		}
		//normalize the vector
		for (i = 0; i < nprof; i++)
			rp[i] /= rnorm;

		//take the fine bins and put them into the selection bins
		tct = 0;
		for (iring = 0; iring < nring; iring++)
		{
			is = (iring - 1) * dnring + 1;
			ie = (iring)* dnring;
			ptot = 0.0;
			for (i = is; i <= ie; i++)
				ptot += rp[i];
			p_ring[iring] = ptot;
		}

		//formulative cumulative distribution function
		cp_ring[0] = p_ring[0];
		for (iring = 1; iring < nring; iring++)
			cp_ring[iring] = cp_ring[iring - 1] + p_ring[iring];

		//find the number of particles in each bin
		memset(n_ring, 0, 4 * nprof);
		//n_ring = 0;

		//cout << "jw nstart = " << nstart << endl;
		//cout << "jw ntot = " << ntot << endl;
		for (i = nstart; i < ntot; i++)
		{
			ran = randm();
			j = 1;
			while (ran > cp_ring[j] && j < nring)
				j = j + 1;
			n_ring[j]++;
		}

		tct = 0;
		i = nstart;
		for (iring = 0; iring <= nring; iring++)
		{
			is = (iring - 1) * dnring + 1;
			ie = (iring)* dnring;
			r1 = float(is)*dr + g.rin;
			r2 = float(ie)*dr + g.rin;
			for (j = 0; j < n_ring[iring]; j++)
			{
				ran = randm();
				r[i] = r1 + ran * (r2 - r1);
				i++;
			}
		}

		//set the angular positions and orbital velocities
		for (i = nstart; i < ntot; i++)
		{
			angle[i] = 2.0 * M_PI * randm();
			v[i] = circular_velocity(g, r[i], p->potential_type, p->a0);
		}

		for (i = nstart; i < ntot; i++)
		{
			st = sin(angle[i]);
			ct = cos(angle[i]);

			x = ct*r[i];
			y = st*r[i];
			z = 0.0;

			xv = -v[i] * st;
			yv = v[i] * ct;
			zv = 0.0;

			x2 = x * ctheta + z * stheta;
			y2 = y;
			z2 = -x * stheta + z * ctheta;
			xv2 = xv * ctheta + zv * stheta;
			yv2 = yv;
			zv2 = -xv * stheta + zv * ctheta;

			x3 = x2  * cphi - y2 * sphi;
			y3 = x2  * sphi + y2 * cphi;
			z3 = z2;
			xv3 = xv2 * cphi - yv2 * sphi;
			yv3 = xv2 * sphi + yv2 * cphi;
			zv3 = zv2;

			p->x0[i].r.x = x3;
			p->x0[i].r.y = y3;
			p->x0[i].r.z = z3;
			p->x0[i].v.x = xv3 + randm()*g.heat;
			p->x0[i].v.y = yv3 + randm()*g.heat;
			p->x0[i].v.z = zv3 + randm()*g.heat;

			//cout << i << "--- " << v[i] << " - " << p->x0[i].print() << endl;
		}
		//delete[] rp, r, angle, v, p_ring, cp_ring, n_ring;
	}
	__device__ double distrb(double r1, double opt, vec rscale)
	{
		if (opt == 1)
			return 1.0 / r1;
		else if (opt == 2)
			return exp(-r1 / rscale.x);
		else if (opt == 3)
			return exp(-r1*r1*rscale.x - rscale.y*r1 - rscale.z);
		return 0;
	}
	__device__ double randm()
	{
		return (double)curand_uniform(&(p->state));
	}
	__device__ double circular_velocity(gparam g, double r, int pot, double a0)
	{
		return circular_velocity(g.mass, r, g.rout, g.eps, pot, a0);
	}
	__device__ double circular_velocity(double mass, double r, double rout, double eps, int pot, double a0)
	{
		double ftotal, tmp;
		int ival;
		if (pot == 0)
			ftotal = mass / (r*r + eps);
		else if (pot == 1)
		{
			ival = df->df_index(r, rout);
			ftotal = mass * df->acceleration_particle[ival] * df->rs2;
		}
		else if (pot == 2)
		{
			ftotal = mass / (r*r + eps);
			tmp = 2 * a0 / ftotal;
			ftotal = ftotal / sqrt(2.0) * sqrt(1.0 + sqrt(1.0 + tmp*tmp));
		}
		return sqrt(ftotal * r);
	}
	__device__ void set_perturber_position(pos_vel pv, double t0, pos_vel*x0, int n1, int n)
	{
		pos_vel xx0;
		int i;
		double tcurrent;

		xx0 = pv;
		tcurrent = t0;

		//now move adjust the test particles from the
		//second disk to the proper velocity and positions

		if (n>n1)
			for (i = n1 + 1; i < n; n++)
				x0[i] = x0[i] + xx0;

		//include the perturbing galaxy
		n++;
		x0[n] = xx0;
	}
	__device__ coe rvToCoe(pos_vel pv, double mu)
	{
		vec h, n, v1, v2, ev;
		vec k(0, 0, 1);
		coe r;
		double muInv, rmag, vmag, hmag, nmag, tmp1, tmp2, p, ecc, cosi, cosO, cosw, cosv, cosu;
		muInv = 1.0 / mu;

		rmag = pv.r.dist();
		vmag = pv.v.dist();

		h = pv.r.cross(pv.v);
		hmag = h.dist();

		n = k.cross(h);
		nmag = n.dist();

		tmp1 = vmag*vmag - mu / rmag;
		tmp2 = pv.r.dot(pv.v);

		v1 = pv.r.scale(tmp1);
		v2 = pv.v.scale(tmp2);

		ev = v1 - v2;
		ev = ev.scale(muInv);

		p = hmag*hmag*muInv;
		ecc = ev.dist();
		cosi = h.z / hmag;
		cosO = n.x / nmag;
		cosw = n.dot(ev) / (nmag*ecc);
		cosv = ev.dot(pv.r) / (ecc*rmag);
		cosu = n.dot(pv.r) / (nmag*rmag);

		r.e = p;
		r.a = ecc;
		r.i = acos(cosi);

		tmp1 = acos(cosO);

		if (n.x < 0)
			tmp1 = 2.0*M_PI - tmp1;

		r.o = tmp1;

		tmp1 = acos(cosw);

		if (ev.y < 0)
			tmp1 = 2.0*M_PI - tmp1;

		r.w = tmp1;

		tmp1 = acos(cosv);

		if (pv.r.dot(pv.v) < 0)
			tmp1 = 2.0*M_PI - tmp1;

		r.v = tmp1;
		return r;
	}
	//* Find the time of rmin, assuming earlier than now, given
	//* the r and v values.Returns r and v at time of rmin
	//* by replacing r and v.r and v are given as
	//* {rx, ry, rz, vx, vy, vz}.
	__device__ tmins getTStart(pos_vel rv, double tmin, double mind)
	{
		double t, distOld, distNew, mu, ecc, a, period, apocenter, a2, tApp, distNearApp;
		double minDist, minVel, xxe7, rv7;
		vec r, v;
		pos_vel tmprv, xxe, rvmin;
		tmins outStuff;
		coe coe;
		bool isEllipse;
		mu = p->galaxy1.mass + p->galaxy2.mass;
		t = 0;
		tmprv = rv;
		tmprv.v = -tmprv.v;
		coe = rvToCoe(tmprv, mu);
		ecc = coe.a;
		a = coe.e / (1.0 - ecc*ecc);
		period = 0.0;
		apocenter = a*(1 + ecc);
		a2 = apocenter*apocenter;
		tApp = 0.0;

		isEllipse = false;

		if (ecc < 1.0)
		{
			isEllipse = true;
			period = 2.0 * M_PI / sqrt(mu)*(pow(a, 1.5));
			period = period * 1.0;
		}


		rvmin = rv; //should assign rvmin the values of rv
		rv7 = 0;
		distNew = rv.r.sqrd();
		distOld = 2.0*distNew;

		distNearApp = -1e30;

		//keep looping as long as distance is decreasing
		while (tmin < t)
		{
			coe = rvToCoe(rv, mu);
			xxe7 = t + p->h;
			wrap_rk41(rv, xxe);

			distNew = xxe.r.sqrd();

			//if it's ellipse and it's near apocenter, take this time
			if (isEllipse && (abs(distNew - a2) / a2 < 0.05))
				if (distNew > distNearApp)
				{
					distNearApp = distNew;
					tApp = t;
				}

			if (distNew < distOld)
			{
				distOld = distNew;
				rvmin = xxe;
				rv7 -= p->h;
			}

			rvmin = xxe;
			rv7 = xxe7 - p->h * 2.0;
			t = t - p->h;
		}
		rv = rvmin;

		minDist = rv.r.dist();
		minVel = rv.v.dist();
		t = rv7;

		if (isEllipse && tApp < 0.0)
			t = tApp;
		else
			t = t - mind / minVel;

		outStuff.t = t;
		outStuff.min_dist = minDist;
		outStuff.min_vel = minVel;
		outStuff.rv7 = rv7;

		return outStuff;
	}
};
class integrator {
public:
	pos_vel xe[Gn], f[Gn], x[Gn];
	double r22[Gn], r21[Gn], r2n[Gn];
	double r1[Gn], r2[Gn], rn[Gn];
	double a1[Gn], a2[Gn], a3[Gn];
	double mond_tmp[Gn];

	double m1, m2, m3;
	double eps1, eps2;
	double theta_i1, phi_i1, theta_i2, phi_i2;
	vec rscale_i1, rscale_i2;
	double rrout1, rrout2;

	double df_force11[Gn], df_force22[Gn], df_forcen[Gn], c3n[Gn];
	int ival11[Gn], ival22[Gn], ivaln[Gn];

	int pn, pn1, pn2;

	parameters* p;

	df_module *df;

	__device__ void init_rkvar()
	{
		int n;

		pn = p->n;
		pn1 = p->galaxy1.n;
		pn2 = p->galaxy2.n;

		n = pn + 1;

		//x = new pos_vel[n];
		//f = new pos_vel[n];
		//xe = new pos_vel[n];

		//r22 = new double[n];
		//r21 = new double[n];
		//r2n = new double[n];
		//r1 = new double[n];
		//r2 = new double[n];
		//rn = new double[n];
		//a1 = new double[n];
		//a2 = new double[n];
		//a3 = new double[n];
		//mond_tmp = new double[n];

		m1 = p->galaxy1.mass;
		m2 = p->galaxy2.mass;
		m3 = p->galaxy2.mass;

		eps1 = p->galaxy1.eps;
		eps2 = p->galaxy2.eps;

		//ival11 = new int[n];
		//ival22 = new int[n];
		//ivaln = new int[n];
		//df_force11 = new double[n];
		//df_force22 = new double[n];
		//df_forcen = new double[n];
		//c3n = new double[n];

		phi_i1 = p->galaxy1.phi;
		theta_i1 = p->galaxy1.theta;
		phi_i2 = p->galaxy2.phi;
		theta_i2 = p->galaxy2.theta;

		rscale_i1 = p->galaxy1.rscale;
		rscale_i2 = p->galaxy2.rscale;

		rrout1 = p->galaxy1.rout;
		rrout2 = p->galaxy2.rout;
	}
	void deallocate_rkvar()
	{
		delete x, f, xe;
		delete r22, r21, r2n;
		delete r1, r2, rn;
		delete a1, a2, a3;
		delete mond_tmp;
	}
	__device__ void wrap_rk4()
	{
		if (p->potential_type == 0) {
			rk4(&integrator::diffeq_spm);
		}
		else if (p->potential_type == 1)
			rk4(&integrator::diffeq_nbi);
		else if (p->potential_type == 2)
			rk4(&integrator::diffeq_mond);
	}
	// -------------------------------------------------- -
	// Use this method so that neither the caller of rk4
	// nor implementation need to know which potential
	// is being used
	// -------------------------------------------------- -
	__device__ void rk4(void(integrator::*diffeq)(pos_vel*))
	{
		int n;
		n = p->n;
		memcpy(x, p->x0, sizeof(pos_vel)*n);

		(this->*diffeq)(x);
		//    cout << "lamb!" << endl;
		for (int i = 0; i < n; i++) {
			/*      cout << xe[i].print() << "xe kitty" << endl;
			cout << x[i].print()  <<  "x kitty" << endl;
			cout << f[i].print() << endl;
			*/
			xe[i] = p->x0[i] + f[i] * p->h / 6.0;
			x[i] = p->x0[i] + f[i] * p->h / 2.0;
			/*
			cout << xe[i].print() << "xe kitty poop" << endl;
			cout << x[i].print()  << "x kitty poop" << endl;*/
		}

		(this->*diffeq)(x);
		for (int i = 0; i < n; i++) {
			xe[i] = xe[i] + f[i] * p->h / 3.0;
			x[i] = p->x0[i] + f[i] * p->h / 2.0;
		}

		(this->*diffeq)(x);
		for (int i = 0; i < n; i++) {
			xe[i] = xe[i] + f[i] * p->h / 3.0;
			x[i] = p->x0[i] + f[i] * p->h;
		}

		(this->*diffeq)(x);
		for (int i = 0; i < n; i++) {
			xe[i] = xe[i] + f[i] * p->h / 6.0;
		}

		memcpy(p->xout, xe, sizeof(pos_vel)*n);
		//p->xout = xe;
	}
	__device__ void diffeq_spm(pos_vel *x)
	{
		pos_vel xn;
		int n;

		n = p->n;
		xn = x[n - 1];
		//cerr << x[n].print() << endl;

		for (int i = 0; i < n; i++) {
			r22[i] = pow((x[i].r.x - xn.r.x), 2) + pow((x[i].r.y - xn.r.y), 2) + pow((x[i].r.z - xn.r.z), 2);
			r21[i] = x[i].r.sqrd();
			r2n[i] = xn.r.sqrd();

			r2[i] = sqrt(r22[i]);
			r1[i] = sqrt(r21[i]);
			rn[i] = sqrt(r2n[i]);

			//cout << x[i].print() << endl;
			//cout << x[i].r.x << "duck " << endl;
			//cout << xn.r.y << endl;

			// this is a correction to prevent NaN errors in the vectorized
			// function evalution at the location of the second mass
			r2[n - 1] = 1.0;

			a1[i] = -m1 / (r21[i] + p->galaxy1.eps);
			a2[i] = -m2 / (r22[i] + p->galaxy2.eps);
			a3[i] = -m3 / (r2n[i] + p->galaxy2.eps);

			// calculate the RHS of the diffeq
			f[i].r = x[i].v;

			f[i].v = x[i].r * a1[i] / r1[i] + (x[i].r - xn.r) * a2[i] / r2[i] + xn.r * a3[i] / rn[i];
			//cout << "acc2 = original" << f[i].print() << endl;
			//f[i].v.x = x[i].r.x * a1[i] / r1[i] + (x[i].r.x - xn.r.x) * a2[i] / r2[i] + xn.r.x * a3[i] / rn[i];
			//f[i].v.y = x[i].r.y * a1[i] / r1[i] + (x[i].r.y - xn.r.y) * a2[i] / r2[i] + xn.r.y * a3[i] / rn[i];
			//f[i].v.z = x[i].r.z * a1[i] / r1[i] + (x[i].r.z - xn.r.z) * a2[i] / r2[i] + xn.r.z * a3[i] / rn[i];
			//cout << "acc2 = explicit" << f[i].print() << endl;

			//cout << "acc " <<  a1[i] << ", " << a2[i] << ", " << a3[i] << endl;
			///cout << "acc2 " << f[i].print() << endl;
			//cout << "acc3 " << x[i].r.print() << endl;
			//cout << "acc4 " << xn.r.print() << endl;
			//cout << "acc5 " << r1[i] << ", " << r2[i] << ", " << rn[i] << endl;
			/*      f[i].r.x = 0;
			f[i].r.y = 0;
			f[i].r.z = 0;

			f[i].v.x = 0;
			f[i].v.y = 0;
			f[i].v.z = 0;
			*/

			//cerr << f[i].v.print() << endl;
			//cerr << r1[i] << " " << r2[i] << " " << rn[i] << endl;
		}
	}
	__device__ void diffeq_nbi(pos_vel *x)
	{
		pos_vel xn;
		int n;

		double df_sigma, df_rho;
		double c1, c2, xvalue, v1, v21;
		double sqrtpi;

		sqrtpi = sqrt(M_PI);

		n = p->n;
		xn = x[n];

		for (int i = 0; i < n; i++) {
			// distance between the main galaxy and the particle
			r21[i] = x[i].r.sqrd();
			r1[i] = sqrt(r21[i]);

			// distance between the companion and the particle
			r22[i] = pow((x[i].r.x - xn.r.x), 2) + pow((x[i].r.y - xn.r.y), 2) + pow((x[i].r.z - xn.r.z), 2);
			r2[i] = sqrt(r22[i]);

			// distance between the two galaxies - the tidal force
			r2n[i] = xn.r.sqrd();
			rn[i] = sqrt(r2n[i]);

			ival11[i] = df->df_index(r1[i], rrout1);
			ival22[i] = df->df_index(r2[i], rrout2);
			ivaln[i] = df->df_index(rn[i], rrout2);

			df_force11[i] = df->acceleration_particle[ival11[i]] * df->rs_internal * df->rs_internal;
			df_force22[i] = df->acceleration_particle[ival22[i]] * df->rs_internal * df->rs_internal;
			df_forcen[i] = df->acceleration_particle[ivaln[i]] * df->rs_internal * df->rs_internal;

			// get the forces, sigma and rho, and rescale them
			df_sigma = df->new_vr2[ivaln[1]] * df->rs_internal * df->rs_internal;
			df_rho = df->new_rho[ivaln[1]] * (df->rs_internal * df->rs_internal * df->rs_internal);

			// interpolated forces 
			a1[i] = -m1 * df_force11[i];
			a2[i] = -m2 * df_force22[i];
			a3[i] = -m3 * df_forcen[i];

		}
		// df
		v21 = xn.v.sqrd();
		v1 = sqrt(v21);

		xvalue = v1 / df_sigma;
		c1 = erf(xvalue) - 2.0 * xvalue / sqrtpi * exp(-xvalue*xvalue);

		// df formula with G=1
		c2 = 4.0 * M_PI * m2 * df->lnl / v21;
		memset(c3n, 0, sizeof(double)*n);
		for (int i = pn1; i < n; i++) {
			c3n[i] = c1 * c2 * df_rho;
		}

		// this is a correction to prevent NaN errors in the vectorized
		// function evalution at the location of the second mass
		r2[n] = 1.0;

		// calculate the RHS of the diffeq
		for (int i = 0; i < n; i++) {
			f[i].r = x[i].v;

			f[i].v = x[i].r * a1[i] / r1[i] + (x[i].r - xn.r) * a2[i] / r2[i] + xn.r * a3[i] / rn[i] - xn.v * c3n[i] / v1;
		}
	}
	__device__ void diffeq_mond(pos_vel *x)
	{
		pos_vel xn;
		int n;

		n = p->n;
		xn = x[n];

		for (int i = 0; i < n; i++) {
			r22[i] = pow((x[i].r.x - xn.r.x), 2) + pow((x[i].r.y - xn.r.y), 2) + pow((x[i].r.z - xn.r.z), 2);
			r21[i] = x[i].r.sqrd();
			r2n[i] = xn.r.sqrd();

			r2[i] = sqrt(r22[i]);
			r1[i] = sqrt(r21[i]);
			rn[i] = sqrt(r2n[i]);

			a1[i] = -m1 / (r21[i] + p->galaxy1.epsilon);
			a2[i] = -m2 / (r22[i] + p->galaxy2.epsilon);
			a3[i] = -m3 / (r2n[i] + p->galaxy2.epsilon);

			// this is a correction to prevent NaN errors in the vectorized
			// function evalution at the location of the second mass
			r2[n] = 1.0;


			// scale the accelerations to reflect mond

			mond_tmp[i] = 2 * p->a0 / a1[i];
			a1[i] = a1[i] / sqrt(2.0) * sqrt(1.0 + sqrt(1.0 + mond_tmp[i] * mond_tmp[i]));

			mond_tmp[i] = 2 * p->a0 / a2[i];
			a2[i] = a2[i] / sqrt(2.0) * sqrt(1.0 + sqrt(1.0 + mond_tmp[i] * mond_tmp[i]));

			mond_tmp[i] = 2 * p->a0 / a3[i];
			a3[i] = a3[i] / sqrt(2.0) * sqrt(1.0 + sqrt(1.0 + mond_tmp[i] * mond_tmp[i]));


			// calculate the RHS of the diffeq

			f[i].r = x[i].v;
			f[i].v = x[i].r * a1[i] / r1[i] + (x[i].r - xn.r) * a2[i] / r2[i] + xn.r * a3[i] / rn[i];
		}
	}

};
class init_module
{
public:
	parameters p;
	setup_module s;
	df_module df;
	integrator in;

	bool header_on;
	vec projected[Gn];
	pos_vel original_rv;
	string fname;

	int argc;
	char** argv;

	__device__ void default_parameters()
	{
		p.potential_type = 0;
		p.standard_galaxy_both();
		p.test_collision();

		custom_collision();

		//p.x0 = new pos_vel[p.n + 1];
		//p.xout = new pos_vel[p.n + 1];
		//projected = new vec[p.n + 1];
	}
	void print_run()
	{
		p.print_profile(1);
		p.print_profile(2);
		p.print_collision();
	}
	__device__ void create_collision()
	{
		double tmpT;
		pos_vel r4min;
		tmins t;

		df.init_distribution();
		// create the disks
		s.profile();
		// determine if we need to calculate tStart
		if (!p.tIsSet)
		{
			r4min.r = p.sec_vec.r;
			r4min.v = -p.sec_vec.v;
			t = s.getTStart(r4min, -30.0, 10.0*p.galaxy1.rout);

			tmpT = t.t;
			if (tmpT < 12.0)
				tmpT = -5;

			if (abs(tmpT) < p.h)
				tmpT = -5;

			p.tstart = tmpT;
			p.time = p.tstart;
			p.tIsSet = true;
		}

		//set the perturber galaxy position
		if (!p.use_sec_vec)
			s.perturber_position(original_rv);
		else
			s.perturber_position_vec(p.sec_vec, original_rv);
	}
	void create_images()
	{
		p.iout++;
		fname = "a." + to_string(p.iout);
		FILE *fp;
		fp = fopen(fname.c_str(), "w+");
		p.output_particles(fp, header_on);
		fclose(fp);
	}
	__device__ void take_a_step()
	{
		p.h = p.hbase;
		in.wrap_rk4();

		//cout << p.x0[0].print() << endl;
		//cout << p.xout[0].print() << endl;


		memcpy(p.x0, p.xout, sizeof(pos_vel)*Gn);
		//p.x0 = p.xout;
		p.time += p.h;
	}
	__device__ void custom_collision()
	{
		p.tIsSet = false;
		//string shortbuff;

		//// If command line arguments were passed, set them here
		//int narg = argc;
		//if (narg > 1)
		//{
		//	cout << "custom collision ---------";
		//	shortbuff = argv[1];
		//	if (shortbuff.compare("-f") == 0)
		//	{
		//		//grab the filename
		//		shortbuff = argv[2];
		//		ifstream ifs(shortbuff);
		//		p.read_parameter_file(ifs);
		//		ifs.close();
		//	}
		//	else
		//	{
		//		p.parse_state_info_string(shortbuff);
		//		p.potential_type = 1;
		//		p.h = p.hbase;
		//		p.tstart = -5;
		//		p.tend = 0;
		//		p.time = -5;
		//		if (narg > 2)
		//		{
		//			shortbuff = argv[2];
		//			p.tstart = stod(shortbuff);
		//			p.time = p.tstart;
		//			p.tIsSet = true;
		//		}
		//	}
		//}
		//else
		//{
		p.galaxy1.phi = 5.0;
		p.galaxy1.theta = 5.0;
		p.galaxy1.rscale = 1.0;
		p.galaxy1.rout = 1.0;
		p.galaxy1.mass = 1.0;
		p.galaxy1.epsilon = 0.3;
		p.galaxy1.n = G1n;
		p.galaxy1.heat = 0.0;
		p.galaxy1.opt = 1;

		p.galaxy2.phi = 0.0;
		p.galaxy2.theta = 0.0;
		p.galaxy2.rscale = 0.30;
		p.galaxy2.rout = 0.5;
		p.galaxy2.mass = 0.5;
		p.galaxy2.epsilon = 0.3;
		p.galaxy2.n = G2n;
		p.galaxy2.heat = 0.0;
		p.galaxy2.opt = 1;

		p.inclination_degree = 20.0;
		p.omega_degree = 0.0;
		p.rmin = 0.90;
		p.velocity_factor = 0.90;

		p.h = p.hbase;
		p.time = -5;
		p.tstart = p.time;
		p.tIsSet = true;

		//}

		p.n = p.galaxy1.n + p.galaxy2.n;
		p.galaxy1.eps = p.galaxy1.epsilon*p.galaxy1.epsilon;
		p.galaxy2.eps = p.galaxy2.epsilon*p.galaxy2.epsilon;
	}
	__device__ vec rotation_vector(double theta, double phi)
	{
		vec in;

		double stheta, ctheta, sphi, cphi;

		stheta = sin(theta * M_PI / 180.0);
		ctheta = cos(theta * M_PI / 180.0);
		sphi = sin(phi * M_PI / 180.0);
		cphi = cos(phi * M_PI / 180.0);

		in.x = 0.0;
		in.y = 0.0;
		in.z = 1.0;

		return p.rotate_frame(in, stheta, ctheta, sphi, cphi);
	}
	//  omitted cross_product and rotate_position... unused... along with rotation_vector
};
class basic_run
{
public:
	init_module init;
	double t0, time_interval;
	int nstep_local;

	__device__ void start(int i, int j) {
		// set the disk parameters
		//srand(time(NULL));
		//curand_init(clock64(), i, 0, &init.p.state);

		init.s.t0 = &t0;
		init.s.p = &init.p;
		init.s.df = &init.df;
		init.in.df = &init.df;
		init.in.p = &init.p;

		// set the target parameters
		init.default_parameters();

		double infos[] = { 070, 1.0,27,27 - 1.76016,-0.40892,-4.48686,-0.2624,-0.29988,-0.53499,0.6195,0.25405,0.75273,0.45032,88.9,91.3,334.8,0.0,0.3,0.3,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0,0,0.94579,-5.23999,0.76974,0.0,1.0,1.0,0.0,0.0,0.0 };
		init.p.set_state_info(infos);
		init.p.galaxy1.theta = theta_min + theta_step + i;
		init.p.galaxy1.phi = phi_min + phi_step + j;
		//init.p.galaxy1.theta = theta_min + (theta_max - theta_min)*curand_uniform(&(init.p.state));
		//init.p.galaxy1.phi = phi_min + (phi_max - phi_min)*curand_uniform(&(init.p.state));
		init.create_collision();



		//
		//---- - loop over the system for the output
		//

		//initialize rk routine
		init.in.init_rkvar();

		t0 = init.p.tstart;

		init.p.nstep = (int)((init.p.tend - t0) / init.p.h) + 2;
		nstep_local = init.p.nstep;

		time_interval = (init.p.tend - t0) * 2;

		//init.p.octave_parameters_out(init.original_rv, init.p.x0[init.p.n]);

		//cout << init.original_rv.print() << endl;

		//FILE *fp1 = fopen("initial.txt", "w+");
		//init.p.output_particles(fp1, init.header_on);
		//fclose(fp1);

		//main integration loop

		init.p.iout = 0;

		//nstep_local = 1;

		for (init.p.istep = 0; init.p.istep < nstep_local; init.p.istep++)
		{
			init.take_a_step();
			if (init.p.istep % 50 == 5)
			{
				//printf("Step:\t%d\n", init.p.istep);
				//cerr << init.p.istep << endl;
			}
		}

		//call CREATE_IMAGES
		//init.fname = "a.101";
		//FILE *fp;
		//fp = fopen(init.fname.c_str(), "w+");
		//init.p.output_particles(fp, init.header_on);
		//fclose(fp);

		//this creates a simple script for animating the output with gnuplot
		//if (!init.header_on)
		//	init.p.create_gnuplot_script();

		//delete init.p.x0;
		//delete init.p.xout;
		//init.in.deallocate_rkvar();
	}
};

//cudaError_t basic_run_cuda(basic_run *runs, unsigned int size);
__global__ void addKernel(basic_run *runs)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int i = threadIdx.x;
	int j = blockIdx.x;
	//runs[i].init.argc = 0;
	runs[index].start(i,j);
	printf("Thread %d:%d,%d Done.\n", index, i, j);
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t basic_run_cuda(basic_run *runs, unsigned int size)
{
	basic_run *dev_runs;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	else
		printf("Set Cuda Device\n");

	// Allocate GPU buffers for runs.
	cudaStatus = cudaMalloc((void**)&dev_runs, size * sizeof(basic_run));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	else
		printf("Allocate runs array\n");

	// Copy run classes from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_runs, runs, size * sizeof(basic_run), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	else
		printf("Copied runs array\n");

	// Launch a kernel on the GPU with one thread for each element.
	printf("Launching Runs\n");
	addKernel <<< 36, 72 >>>(dev_runs);
	printf("Finished Runs\n");

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	else
		printf("No Kernel Errors\n");

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	else
		printf("Cuda Device Syncronized\n");

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(runs, dev_runs, size * sizeof(basic_run), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	else
		printf("Copied runs array back\n");

Error:
	cudaFree(dev_runs);

	return cudaStatus;
}

int main()
{
	clock_t begin = clock();

	const int arraySize = 36*72;
	basic_run *runs = new basic_run[arraySize];

	// Add vectors in parallel.
	cudaError_t cudaStatus = basic_run_cuda(runs, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "basic_run_cuda failed!");
		system("pause");
		return 1;
	}
	else
		printf("Kernel Run Success\n");

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		system("pause");
		return 1;
	}

	printf("Writing Output Files...");
	for (int i = 0; i < arraySize; i++)
	{
		//call CREATE_IMAGES
		runs[i].init.fname = "a." + to_string(i);
		FILE *fp;
		fp = fopen(runs[i].init.fname.c_str(), "w+");
		runs[i].init.p.output_particles(fp, true);
		fclose(fp);
	}
	printf(" Done\n");

	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("Time: %f\n", time_spent);
	system("pause");
	return 0;
}
