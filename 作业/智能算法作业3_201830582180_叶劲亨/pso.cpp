#include <iostream>
#include <string>
#include <vector>
#include <deque>
#include <random>
#include <algorithm>
#include <fstream>
#include <time.h>
#include <math.h>

using namespace std;
#define PI acos(-1)

class F {
public:
	double operator()(const vector<double>& x) {
		double res = 0.0;
		for (double d : x) {
			res += d * d - 10 * cos(2 * PI * d) + 10;
		}
		return res;
	}
};

class PSO {
public:
	F func;
	//种群规模
	int pop_size;
	//惯性权重
	double w;
	//迭代次数
	int generations;
	//加速因子
	double c1;
	double c2;
	//粒子长度
	int particle_len;
	//取值范围
	vector<vector<double>>* particles_bound;		// 30 * 2
	//粒子
	vector<vector<double>> particles;				// 20 * 30

	//每个粒子的速度
	vector<vector<double>> velocities;

	//每个粒子的最优位置
	vector<vector<double>> p_best;
	vector<double> p_best_val;
	//全局最优
	vector<double> g_best;
	double g_best_val = INT_MAX;

public:
	PSO(F func, int pop_size, double w, int generations, double c1, double c2, int particle_len, vector<vector<double>>* particles_bound) :
		func(func), pop_size(pop_size), w(w), generations(generations), c1(c1), c2(c2),
		particle_len(particle_len), particles_bound(particles_bound) {}

	void init_particles() {
		this->particles = vector<vector<double>>(this->pop_size);
		this->p_best = vector<vector<double>>(this->pop_size);
		this->p_best_val = vector<double>(this->particle_len, INT_MAX);
		//初始化速度
		this->velocities = vector<vector<double>>(this->pop_size);

		for (int i = 0; i < pop_size; ++i) {
			for (int j = 0; j < particle_len; ++j) {
				double span = (*particles_bound)[i][1] - (*particles_bound)[i][0];
				particles[i].push_back(rand() / (double)(RAND_MAX)* span + (*particles_bound)[i][0]);
				velocities[i].push_back(rand() / (double)(RAND_MAX) * 20 - 10);
			}
		}
	}

	void evaluate() {
		g_best.clear();
		double temp;
		for (int i = 0; i < pop_size; ++i) {
			temp = func(particles[i]);
			//cout << temp << endl;
			if (temp < p_best_val[i]) {
				p_best_val[i] = temp;
				p_best[i] = vector<double>(particles[i]);
			}
			if (temp < g_best_val) {
				g_best_val = temp;
				this->g_best = vector<double>(particles[i]);
			}
		}
	}

	void update() {
		double span = 10.24;
		double v_span = span / 5;
		double temp;
		for (int i = 0; i < pop_size; ++i) {
			for (int j = 0; j < particle_len; ++j) {
				double& p = particles[i][j];
				double& v = velocities[i][j];
				v = w * v + c1 * ((double)rand() / RAND_MAX) * (p_best[i][j] - p) + c2 * ((double)rand() / RAND_MAX) * (g_best[j] - p);
				if (v > v_span) {
					v = v > 0 ? v_span : -v_span;
				}

				p += velocities[i][j];
				if (p < (*particles_bound)[j][0] + 1e-5 || p >(*particles_bound)[j][1] - 1e-5) {
					p = rand() / (double)(RAND_MAX)* span + (*particles_bound)[i][0];
				}
			}
			//异步更新
			temp = func(particles[i]);
			if (temp < p_best_val[i]) {
				p_best_val[i] = temp;
				p_best[i] = vector<double>(particles[i]);
			}
			if (temp < g_best_val) {
				g_best_val = temp;
				this->g_best = vector<double>(particles[i]);
			}
		}
	}

	void run() {
		this->init_particles();
		evaluate();
		for (int i = 0; i < generations; ++i) {
			//cout << "generation: " << i << ": " << g_best_val << endl;
			update();
		}
		//cout << "finished: " << g_best_val << endl;
	}


	void test() {
		for (int i = 0; i < pop_size; ++i) {
			for (int j = 0; j < particle_len; ++j) {
				cout << particles[i][j] << " ";
			}
			cout << endl;
		}
	}


};

void writeHead(ofstream& file, int p_size) {
	file << "w,c1=c2,epoch,g_best\n";
	//for (int i = 0; i < p_size; ++i) {
	//	file << "," << "x" << i;
	//}
	//file << "\n";
}


int main() {
	F func();
	int pop_size = 20;
	double w = 0.729;
	int generations = 2000;
	double c1 = 1.49445;
	double c2 = 1.49445;
	int particle_len = 30;
	vector<vector<double>>* bound = new vector<vector<double>>(30);
	for (int i = 0; i < 30; ++i) {
		(*bound)[i].push_back(-5.12);
		(*bound)[i].push_back(5.12);
	}

	vector<double> w_list({ 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 });
	vector<double> c_list({ 0.5,1.0,1.5,2.0,2.5,3.0,3.5 });
	int epochs = 30;
	//PSO pso(F(), pop_size, w, generations, c1, c2, particle_len, bound);
	//pso.run();
	//cout << pso.g_best_val << endl;

	ofstream file;
	file.open("D://data.csv");

	writeHead(file, particle_len);

	int start = clock();
	for (double w1 : w_list) {
		for (double c : c_list) {
			for (int e = 0; e < epochs; ++e) {
				PSO pso(F(), pop_size, w1, generations, c, c, particle_len, bound);
				pso.run();
				cout << "w: " << w1 << ", c1=c2: " << c << ", epoch: " << e << ", g_best: " << pso.g_best_val << endl;
				/*for (int i = 0; i < 30; i++) {
					cout << pso.g_best[i] << " ";
				}*/
				/*if (pso.g_best_val > 10000 - 1e-5) {
					cout << "************" << endl;
					for (int i = 0; i < pop_size; ++i) {
						cout << "pop" << i << endl;
						for (int j = 0; j < particle_len; ++j) {
							cout << pso.particles[i][j] << " ";
						}
						cout << endl;
					}
					cout << endl;
				}
				cout << endl;*/
				file << w1 << "," << c << "," << e << "," << pso.g_best_val;
				/*for (double p : pso.g_best) {
					file << "," << p;
				}*/
				file << "\n";
			}
		}
	}
	cout << "duration: " << clock() - start << endl;

}