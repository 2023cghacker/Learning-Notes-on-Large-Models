import numpy as np
import random
import time


class GridWorld:
    """网格世界环境"""

    def __init__(self, size=5):
        self.size = size  # 网格大小
        self.grid = np.zeros((size, size))

        # 设置障碍物
        self.obstacles = [(1, 1), (1, 2), (2, 4), (3, 2)]
        for x, y in self.obstacles:
            self.grid[x, y] = -1  # 障碍物标记为-1

        # 设置起点和终点
        self.start = (0, 0)
        self.end = (4, 4)
        self.grid[self.start] = 0  # 起点
        self.grid[self.end] = 10  # 终点奖励

        # 当前位置
        self.current_pos = self.start

    def reset(self):
        """重置环境，回到起点"""
        self.current_pos = self.start
        return self.current_pos

    def step(self, action):
        """执行动作并返回新状态、奖励和是否结束"""
        x, y = self.current_pos

        # 定义动作: 上、右、下、左
        if action == 0:  # 上
            new_x, new_y = max(0, x - 1), y
        elif action == 1:  # 右
            new_x, new_y = x, min(self.size - 1, y + 1)
        elif action == 2:  # 下
            new_x, new_y = min(self.size - 1, x + 1), y
        else:  # 左
            new_x, new_y = x, max(0, y - 1)

        # 检查是否撞到障碍物
        if (new_x, new_y) in self.obstacles:
            return self.current_pos, -5, False  # 撞到障碍物，给予惩罚

        # 更新位置
        self.current_pos = (new_x, new_y)

        # 检查是否到达终点
        if self.current_pos == self.end:
            return self.current_pos, self.grid[self.end], True  # 到达终点，给予奖励

        # 普通步骤，小惩罚鼓励尽快到达终点
        return self.current_pos, -0.1, False

    def render(self):
        """可视化网格世界"""
        for i in range(self.size):
            row = []
            for j in range(self.size):
                if (i, j) == self.current_pos:
                    row.append("A")  # 智能体
                elif (i, j) == self.end:
                    row.append("T")  # 终点
                elif (i, j) in self.obstacles:
                    row.append("#")  # 障碍物
                else:
                    row.append(".")  # 空地
            print(" ".join(row))
        print()


class QLearningAgent:
    """Q-Learning智能体"""

    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.env = env
        self.learning_rate = learning_rate  # 学习率
        self.discount_factor = discount_factor  # 折扣因子
        self.epsilon = epsilon  # 探索率

        # 初始化Q表: [size x size x 4]，4个动作
        self.q_table = np.zeros((env.size, env.size, 4))

    def choose_action(self, state):
        """根据ε-贪婪策略选择动作"""
        x, y = state
        if random.uniform(0, 1) < self.epsilon:
            # 探索: 随机选择动作
            return random.choice(range(4))
        else:
            # 利用: 选择Q值最大的动作
            return np.argmax(self.q_table[x, y, :])

    def learn(self, state, action, reward, next_state):
        """学习并更新Q表"""
        x, y = state
        next_x, next_y = next_state

        # Q-Learning更新公式
        current_q = self.q_table[x, y, action]
        next_max_q = np.max(self.q_table[next_x, next_y, :])
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q
        )

        # 更新Q值
        self.q_table[x, y, action] = new_q


def train_agent(episodes=10000):
    """训练智能体"""
    # 创建环境和智能体
    env = GridWorld(size=5)
    agent = QLearningAgent(env)

    # 记录每个episode的总奖励
    rewards_per_episode = []

    print("开始训练...")
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # 选择动作
            action = agent.choose_action(state)

            # 执行动作
            next_state, reward, done = env.step(action)

            # 学习更新
            agent.learn(state, action, reward, next_state)

            total_reward += reward
            state = next_state

        rewards_per_episode.append(total_reward)

        # 每1000个episode打印一次进度
        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(
                f"Episode {episode+1}/{episodes}, 最近100次平均奖励: {avg_reward:.2f}"
            )

    print("训练完成!")
    return env, agent


def test_agent(env, agent, episodes=5):
    """测试训练好的智能体"""
    print("\n开始测试...")
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        print(f"\n测试 Episode {episode+1}:")
        env.render()
        time.sleep(0.5)

        while not done and steps < 50:  # 限制最大步数防止无限循环
            # 选择最优动作（不探索）
            action = np.argmax(agent.q_table[state[0], state[1], :])

            # 执行动作
            next_state, reward, done = env.step(action)

            total_reward += reward
            state = next_state
            steps += 1

            # 显示当前状态
            env.render()
            time.sleep(0.5)

        print(
            f"测试 Episode {episode+1} 完成, 总奖励: {total_reward:.2f}, 步数: {steps}"
        )


if __name__ == "__main__":
    # 训练智能体
    env, agent = train_agent(episodes=10000)

    # 测试智能体
    test_agent(env, agent, episodes=3)

    # 打印最终的Q表（简化版）
    print("\n最终Q表（每个状态的最佳动作值）:")
    for i in range(env.size):
        row = []
        for j in range(env.size):
            if (i, j) in env.obstacles:
                row.append("  #  ")
            else:
                max_q = np.max(agent.q_table[i, j, :])
                row.append(f"{max_q:6.2f}")
        print(" ".join(row))
