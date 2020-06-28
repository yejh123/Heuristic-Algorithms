import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
import time


def plt_function(F):
    x1 = np.linspace(-3.0, 12.1, 50)
    x2 = np.linspace(4.1, 5.8, 50)
    xx, yy = np.meshgrid(x1,x2)

    z = F(xx,yy)
    print(z.max())

    # 绘制3D图像
    ax = plt.figure().add_subplot(111, projection='3d')
    plt.xlabel('x1')
    plt.ylabel('x2')
    ax.scatter(xx.flatten(), yy.flatten(), z, c='r', marker='o', s=1)  # 点为红色三角形


class GA(object):
    # 初始化，F是待求解的函数，DNA_size是F中变量的个数（本次问题为2），DNA_bound：表示变量的取值范围
    # cross_rate：交配概率，mutate_rate：变异概率，pop_size：群体中个体数目
    def __init__(self, F, DNA_size, DNA_bound, cross_rate, mutate_rate, pop_size):
        self.F = F
        self.DNA_size = DNA_size
        self.DNA_bound = DNA_bound
        self.cross_rate = cross_rate
        self.mutate_rate = mutate_rate
        self.pop_size = pop_size

        # init population
        self.population = np.empty((pop_size, DNA_size))
        for i in range(len(DNA_bound)):
            span = self.DNA_bound[i][1] - self.DNA_bound[i][0]
            self.population[:, i] = np.random.rand(self.pop_size) * span + self.DNA_bound[i][0]

    # count how many character matches（计算适应值）
    def get_fitness(self):
        return self.F(self.population[:, 0], self.population[:, 1])

    # 完成选择
    def select(self):
        fitness = self.get_fitness() + 1e-4  # add a small amount to avoid all zero fitness
        # 模拟轮盘赌算法，按fitness的比例选取个体，replace=True表示可放回选取
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness / fitness.sum())
        return self.population[idx]

    # 完成交配
    def crossover(self, parent, pop):
        if np.random.rand() < self.cross_rate:
            i_ = np.random.randint(0, self.pop_size, size=1)  # select another individual from pop
            cross_points = np.random.randint(0, 2, self.DNA_size).astype(np.bool)  # choose crossover points
            parent[cross_points] = pop[i_, cross_points]  # mating and produce one child
        return parent

    # 完成变异
    def mutate(self, child):
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                span = self.DNA_bound[point][1] - self.DNA_bound[point][0]
                child[point] = np.random.rand() * span + DNA_bound[point][0]  # choose a random ASCII index
        return child

    # 完成一次进化
    def evolve(self):
        # 选择select
        pop = self.select()
        # 交配
        original_pop = pop.copy()
        for parent in pop:
            # print("p1", parent)   # parent 可能会改变
            child = self.crossover(parent, original_pop)
            # print("p2", parent)  # parent 可能会改变
            child = self.mutate(child)
            # print("p3", parent)  # parent 可能会改变
            parent[:] = child  # 不写这行也可以

        self.population = pop


if __name__ == '__main__':
    F = lambda x1, x2: 21.5 + x1 * np.sin(4 * np.pi * x1) + x2 * np.sin(20 * np.pi * x2)
    N = 100  # population size
    GMAX = 2000
    pc_list = np.linspace(0.1, 0.9, 9)  # mating probability (DNA crossover)
    pm_list = np.linspace(0.01, 0.09, 9)  # mutation probability
    epochs = 30
    DNA_bound = [(-3.0, 12.1), (4.1, 5.8)]

    res = []
    best_DNA_list = []
    start = time.clock()
    for pc in pc_list:
        for pm in pm_list:
            for epoch in range(epochs):
                # 创建GA对象
                ga = GA(F, DNA_size=2, DNA_bound=DNA_bound, cross_rate=pc, mutate_rate=pm, pop_size=N)
                # pop = ga.population
                best_DNA = []
                best_fitness = 0
                for generation in range(GMAX):
                    fitness = ga.get_fitness()
                    best_idx = np.argmax(fitness)
                    fitness_temp = fitness[best_idx]
                    best_fitness = max(best_fitness, fitness_temp)
                    best_DNA = ga.population[best_idx] if best_fitness == fitness_temp else best_DNA
                    ga.evolve()

                print('pc: {}, pm: {}, epoch: {}, best_DNA: {}, fitness: {}'.format(pc, pm, epoch, best_DNA,
                                                                                    best_fitness))
                res.append(best_fitness)
                best_DNA_list.extend(best_DNA.tolist())

    res = np.array(res).reshape((len(pc_list), len(pm_list), epochs))
    best_DNA_list = np.array(best_DNA_list).reshape((len(pc_list), len(pm_list), epochs, 2))

    df = pd.DataFrame(columns=['pc', 'pm', 'epoch', 'best_fitness', 'x1', 'x2'])
    i = 0
    for pc_idx in range(len(pc_list)):
        for pm_idx in range(len(pm_list)):
            for epoch in range(epochs):
                df.loc[i] = [pc_list[pc_idx], pm_list[pm_idx], epoch, res[pc_idx][pm_idx][epoch], best_DNA_list[pc_idx][pm_idx][epoch][0],
                             best_DNA_list[pc_idx][pm_idx][epoch][1]]
                i += 1

    print('finished, duration: {}'.format(time.clock() - start))

    # 输出文件
    print(df)
    df.to_csv("./data.csv", index=False)

    # 绘图
    res = np.mean(res, axis=2)
    res = res.reshape((len(pc_list), len(pm_list)))
    plt.figure(figsize=(20,10))
    sns.heatmap(res, annot=True, fmt='.3f', xticklabels=[str(pc)[:4] for pc in pc_list], yticklabels=[str(pm)[:4] for pm in pm_list])

